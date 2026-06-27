# Func

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

`ccu::Func`与`ccu::CallFunc<Obj>`共同构成CCU kernel内的FuncBlock机制，用于在kernel内将一段重复使用的逻辑只生成一份指令，多处调用均通过`Call`指令跳转，节省SRAM。

- `ccu::Func`：封装一个lambda为可复用的FuncBlock定义。lambda形参为零个或多个`ccu::Variable`，返回`void`。
- `ccu::CallFunc<Obj>(args...)`：调用点接口。首次调用时合成FuncBlock段并追加一条`Call`指令；后续调用直接追加`Call`指令，复用已有的FuncBlock段。

## 类定义与模板函数

```cpp
namespace AscendC {
namespace ccu {

class Func {
public:
    // Lambda 必须返回void，形参必须全部为ccu::Variable
    template <typename Lambda>
    explicit Func(Lambda body);

    Func(const Func &) = delete;             // 不可复制
    Func &operator=(const Func &) = delete;  // 不可赋值
};

// Obj 必须为global 或static 存储的ccu::Func 对象（C++ NTTP 要求）
// Args 中每个参数必须为ccu::Variable
template <Func &Obj, typename... Args>
CcuResult CallFunc(Args... args);

} // namespace ccu
} // namespace AscendC
```

## Func构造参数说明

| 参数名 | 说明 |
| --- | --- |
| body | lambda表达式，形参为零个或多个`ccu::Variable`，返回`void`。形参个数在编译期由lambda签名自动推导。 |

## CallFunc参数说明

| 参数名 | 说明 |
| --- | --- |
| Obj | 模板非类型参数（NTTP），必须为`global`或`static`存储的`ccu::Func`对象的引用。 |
| args... | 实参列表，个数必须与`Obj`的lambda形参个数一致，每个参数类型必须为`ccu::Variable`。 |

## 返回值（CallFunc）

[CcuResult](../../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 操作成功。 |
| `CCU_E_PARA` | 实参个数与`Obj.NumIn()`不一致（该校验在最前，直接返回）。 |
| `CCU_E_PTR` | 当前不存在处于注册中的kernel（接口在kernel注册阶段之外被调用）。 |

> [!NOTE]说明
> 其余失败情形（如在硬件Loop body 内调用、嵌套CallFunc 等）会抛出异常（携带错误码），不通过返回值反馈（见下节"异常"）。

## 异常（CallFunc）

`CallFunc`调用失败时会抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码），常见情形：

| 情形 | 异常携带的错误码 |
| --- | --- |
| 在硬件Loop body 内调用`CallFunc`，或在`Func`的lambda body内嵌套调用`CallFunc`（仅允许在顶层调用） | `CCU_E_INTERNAL` |
| FuncBlock合成失败 | 对应错误码 |

## 约束说明

- `ccu::Func`对象必须具有"链接性"：只能定义在namespace作用域（含`static`）或作为类的静态数据成员，不能是函数局部变量（即使加`static`也不行）。这是`template <Func& Obj, ...>`引用类型NTTP的C++语言要求，违反时编译期直接报错，与运行时无关。`CallFunc`内部还会用`&Obj`作为标识来复用同一个FuncBlock，这是该约束的副作用收益，不是约束的成因。
- `ccu::Func`不可复制（拷贝构造和赋值运算符已`= delete`），每个Func对象有唯一身份。
- `ccu::Func`lambda形参类型必须能从`ccu::Variable&`传入，实际可写为`ccu::Variable` / `ccu::Variable&` / `const ccu::Variable&`三者之一，其他类型（如`int`、`ccu::Address`）会因模板替换失败而编译报错（注：当前未对形参类型加`static_assert`，错误信息以深层模板诊断形式出现，可读性不如返回值类型那条）。
- `ccu::Func`lambda必须返回`void`，否则编译期报错（由实现内显式`static_assert`给出可读诊断）。
- `CallFunc<F>`不可在硬件Loop body内调用，否则抛出异常（错误码`CCU_E_INTERNAL`）。
- `CallFunc<F>`不可嵌套：lambda内不可再调用其他`CallFunc`，否则抛出异常（错误码`CCU_E_INTERNAL`）；框架在`Func`结束阶段还会再做一次兜底检查，对应上文《异常》表中的"FuncBlock合成失败"行。
- 首次`CallFunc`时合成FuncBlock，对同一`Obj`的后续`CallFunc`仅追加`Call`指令，复用已有FuncBlock。不同的`ccu::Func`对象即使lambda内容相同，也各自合成独立的FuncBlock（Key是`&Obj`，不是lambda内容的hash）。

## 调用示例

```cpp
using namespace AscendC::ccu;

// 声明为static，满足CallFunc NTTP 要求
static Func MyAdd([](Variable x) {
    Variable tmp;
    tmp = x + x;   // 对x 翻倍
});

CcuResult MyKernel(CcuKernelArg arg) {
    Variable a, b;
    a = 2;
    b = 5;

    // 首次调用：合成FuncBlock 段 + 追加Call 指令
    CcuResult ret = CallFunc<MyAdd>(a);
    if (ret != CCU_SUCCESS) { return ret; }

    // 第二次调用：仅追加Call 指令，复用已有FuncBlock
    return CallFunc<MyAdd>(b);
}
```
