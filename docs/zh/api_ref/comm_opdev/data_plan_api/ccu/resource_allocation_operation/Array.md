# Array

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

`ccu::Array<T>`是CCU kernel内批量申请物理连续资源的C++模板类。

- 构造即批量分配：构造时一次性申请`count`个物理连续的虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。
- 仅可移动：禁止拷贝，允许移动。

物理连续是某些接口的前置条件：`Load(addr, vArr, num)`/`Store`批量加载/存储、`LocalReduce(buffers*, count, ...)`（2 ≤ count ≤ 8，详见 [LocalReduce](../data_movement/LocalReduce.md)）都要求参数物理连续，必须通过`Array<T>`申请，单独声明多个对象不保证物理连续。

当前仅支持以下三种特化类型：

| 特化类型 | 对应硬件资源 |
| --- | --- |
| `Array<Variable>` | N个物理连续XN寄存器 |
| `Array<Event>` | N个物理连续CKE完成事件位 |
| `Array<CcuBuffer>` | N个物理连续MS切片 |

对其他类型实例化`Array<T>`将在编译期报错（仅支持上述三种特化类型）。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
template <typename T>  // T 仅支持Variable / Event / CcuBuffer
class Array final {
public:
    explicit Array(uint32_t count);       // 构造即批量虚拟分配，count可为0
    T& operator[](uint32_t i);            // 下标访问（无边界检查）
    const T& operator[](uint32_t i) const;
    T* data();                             // 获取首元素指针（用于传给需要指针参数的批量接口）
    const T* data() const;
    uint32_t size() const;                 // 返回元素个数
    // 禁止拷贝；允许移动
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& other) noexcept;
    Array& operator=(Array&& other) noexcept;
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Array<Variable> vars(n);` | 申请`n`个物理连续XN句柄。 |
| `Array<Event> evts(n);` | 申请`n`个物理连续CKE句柄。 |
| `Array<CcuBuffer> bufs(n);` | 申请`n`个物理连续MS句柄。 |

`count`可以为0（`size() == 0`），此时不分配硬件资源。构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。

## 成员函数说明

| 函数 | 说明 |
| --- | --- |
| `arr[i]` | 返回第`i`个元素的引用（从0起，无边界检查）。 |
| `arr.data()` | 返回首元素指针，用于传给需要`T*`参数的批量接口（如`LocalReduce(bufs.data(), count, ...)`）。 |
| `arr.size()` | 返回申请时的`count`值。 |

## 约束说明

- 只能在kernel注册阶段构造Array。
- 析构不释放硬件资源，不应在kernel之外保存元素的`handle`值，翻译完成后句柄即失效。
- `Array<T>`仅特化`Variable/Event/CcuBuffer`三种类型，对其他类型实例化将在编译期失败。
- 多个单独声明的`Variable`/`Event`/`CcuBuffer`对象不保证物理连续，不可用于需要物理连续资源的接口（如批量`Load`/`Store`、多Buffer`LocalReduce`）。

> [!NOTE]说明
> `Load`/`Store`会对Variable数组做连续性校验（不连续返回`CCU_E_PARA`）；而`LocalReduce`的多Buffer重载不校验CcuBuffer是否物理连续，须由调用方自行保证。用非`Array`申请的多个Buffer时不会立即报错，但运行期行为未定义。

- C++ 构造只申请虚拟句柄，恒成功；资源池无法凑出N个连续物理资源时，在`HcommCcuKernelRegister`阶段返回`CCU_E_UNAVAIL`，不是在构造时抛出。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    // 批量申请4个物理连续Variable，用于Load批量加载
    Array<Variable> vArr(4);
    Load(0x80000000ULL, vArr, 4);    // 一次加载4个uint64_t

    // 批量申请4个物理连续CcuBuffer，用于多Buffer归约
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;
    len = 4096;
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);

    // 下标访问单个元素
    vArr[0] = 1024;    // 对第0个Variable赋立即数

    return CCU_SUCCESS;
}
```
