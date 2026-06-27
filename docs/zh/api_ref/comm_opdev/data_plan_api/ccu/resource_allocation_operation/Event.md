# Event

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

`ccu::Event`是CCU kernel内完成事件位（CKE）的C++包装类。

- 构造即分配：默认构造函数自动申请1个CKE虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。

Event用于标记异步操作（如数据搬运）的完成状态。数据搬运接口的末位参数`(event, mask)`由硬件在搬运完成时自动置位`event[mask]`，下游通过`EventWait(event, mask)`等待该位被置位。Event本身不持有`mask`字段，mask在每次`EventRecord`/`EventWait`调用时独立传入。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
class Event final {
public:
    Event();                    // 构造即Alloc
    CcuEventHandle handle{0};  // 虚拟句柄
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Event e;` | 申请1个CKE虚拟句柄。只能在kernel注册阶段（`HcommCcuKernelRegister`执行的kernel函数体内）调用。 |

> [!CAUTION]注意
> `Event` 类有copy/move构造函数，它们只拷贝`handle`字段、不申请新的CKE——`Event e2 = e1;` 后`e1`和`e2`指向同一个CKE，对任一对象`EventRecord`/`EventWait`操作的是同一份硬件状态位。如需独立Event，必须显式`Event e2;`默认构造（或用`Array<Event>`）。

构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。

## 返回值

构造成功后，`e.handle`字段持有分配到的虚拟句柄。句柄值由框架在注册阶段分配（首个Event 的handle 为 `0`，后续递增），不要用"`handle != 0`"判断有效性。

## 约束说明

- 只能在kernel注册阶段构造Event。
- 析构不释放硬件资源，不应在kernel之外保存或比较`handle`值，翻译完成后句柄即失效。
- Event没有`mask`字段，mask在`EventRecord`/`EventWait`及数据搬运接口中以参数形式传入（C++默认值为`1`）。同一Event的不同bit（mask）相互独立，可承载多组配对。
- C++ 构造只申请虚拟句柄，恒成功不抛异常；`CKE`物理资源不足时，由`HcommCcuKernelRegister`阶段返回`CCU_E_UNAVAIL`，不是在构造时抛出。如需多个Event，可使用[`Array<Event>`](Array.md)批量申请。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;    // 申请1个CKE

    // 配合数据搬运接口使用：硬件搬运完成时自动置位evt[0x1]
    LocalAddr src, dst;
    Variable len;
    LocalCopy(dst, src, len, evt);    // mask默认0x1
    EventWait(evt);                    // 等待bit0置位

    return CCU_SUCCESS;
}
```
