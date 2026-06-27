# Variable

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

`ccu::Variable`是CCU kernel内标量寄存器（XN）的C++包装类。

- 构造即分配：默认构造函数自动申请1个XN虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。
- 运算符即device操作：Variable上的赋值与算术运算符描述的是device端（硬件）执行的操作，运行期操作对应的XN寄存器，而非在host端立即计算。

> [!NOTE]说明
> CCU资源分配采用"先虚后实"两阶段模型：注册阶段的`Variable()`构造仅产生虚拟句柄，真正的物理XN 分配在 `HcommCcuKernelRegister` 阶段（kernel 函数执行完后）完成。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
class Variable final {
public:
    Variable();                                                   // 构造即Alloc
    void operator=(uint64_t immediate) const;                    // 赋立即数
    void operator=(const Variable& other) const;                 // Variable间赋值
    void operator+=(const Variable& other) const;               // 就地加法
    /*内部表达式对象*/ operator+(const Variable& that) const;    // 加法（表达式模板）
    CondExpr operator==(uint64_t immediate);                     // 产出CondExpr
    CondExpr operator!=(uint64_t immediate);                     // 产出CondExpr
    CcuVariableHandle handle{0};                                 // 虚拟句柄
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `Variable v;` | 申请1个XN虚拟句柄。只能在kernel注册阶段（`HcommCcuKernelRegister`执行的kernel函数体内）调用。 |

`Variable` 类有copy/move构造函数（`Variable(const Variable&)` / `Variable(Variable&&)`），它们只拷贝`handle`字段、不申请新的XN——即`Variable v2 = v1;` 后`v1`和`v2`指向同一个硬件寄存器，对任一对象的赋值/算术运算操作的是同一份XN。如需独立的Variable，必须显式`Variable v2;`默认构造。同理，move赋值`v2 = std::move(v1)`也只是handle拷贝（不剥夺源handle）。

这与`operator=(const Variable&)`完全不同——后者会发出device端寄存器赋值指令，操作的是两个独立寄存器之间的值搬移。

构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。

## 运算符说明

### 赋值运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `v = imm;`（`imm`为`uint64_t`） | `XN_v ← imm`。立即数在注册阶段确定，运行期不可变。 |
| `d = s;`（`s`为`Variable`） | `XN_d ← XN_s`。device端寄存器赋值，而非host端handle拷贝。 |

### 算术运算符

| 表达式写法 | 硬件语义 |
| --- | --- |
| `r = a + b;` | `XN_r ← XN_a + XN_b`（单条双源加法指令）。`operator+`返回表达式模板对象，被`operator=`消费时生成一条device加法，不产生临时Variable。 |
| `r += b;` | `XN_r ← XN_r + XN_b`。与`r = r + b`语义相同，无临时对象。 |

> [!CAUTION]注意
> `r = a + b`使用表达式模板（内部类型）以避免产生临时Variable占用额外XN寄存器。不要把`a + b`的结果存入普通C++变量，否则不会生成对应的device操作。

### 条件运算符

| 表达式写法 | 返回类型 | 说明 |
| --- | --- | --- |
| `n == imm` | `CondExpr` | 产出条件表达式对象，不产生任何device操作，专供`CCU_IF`/`CCU_WHILE`/`CCU_DO...CCU_WHILE()`宏消费。 |
| `n != imm` | `CondExpr` | 同上。 |

> [!CAUTION]注意
> `CondExpr`只能被控制流宏消费，不能用作普通C++布尔表达式（如`if (n == 0)`）。塞入普通`if`不会产生任何CCU控制流，表达式被直接丢弃。

## 约束说明

- 只能在kernel注册阶段构造Variable。
- 析构不释放硬件资源，不应在kernel之外保存或比较`handle`值，翻译完成后句柄即失效。
- 当前仅支持加法运算，不支持减法、乘法、除法。
- 立即数不可直接参与算术（`v + 1`不合法），须先赋给一个Variable（`one = 1;`）后再参与运算（`v = v + one;`）。
- 构造仅申请虚拟句柄（不消耗物理 XN），**在 kernel 注册阶段内构造恒成功**；若在注册阶段之外构造（不在 `HcommCcuKernelRegister` 调用的 kernel 函数体内），底层 `CcuVariableAlloc` 找不到当前 kernel，会抛出携带 `CCU_E_PTR` 的 `CcuException`。XN 物理资源不足由 `HcommCcuKernelRegister` 阶段返回 `CCU_E_UNAVAIL`，不在构造时触发。该约束与上文"只能在kernel注册阶段构造Variable"前后呼应——后者讲规则，前者讲违反规则的后果（抛 `CCU_E_PTR`）。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, i, one, step;

    // 赋立即数（注册阶段确定）
    n = 100;
    i = 0;
    one = 1;
    step = 8;

    // Variable间赋值
    Variable cursor;
    cursor = i;           // XN_cursor ← XN_i

    // 算术：表达式模板写法（r = a + b），仅生成一条device加法
    Variable sum;
    sum = i + step;       // XN_sum ← XN_i + XN_step

    // 就地加法
    i += one;             // XN_i ← XN_i + XN_one

    // 条件运算（供CCU_WHILE宏消费）
    CCU_WHILE(n != 0) {
        // ...
    }

    return CCU_SUCCESS;
}
```
