# 简介

本节提供CCU kernel内虚拟资源句柄的申请接口，以及对`Variable`/`Address`进行赋值与算术运算的接口。

CCU资源分配采用"先虚后实"两阶段模型：用户在kernel注册阶段（kernel函数体内）调用 `Variable()`/`Address()`/`Event()` 等构造函数仅产生虚拟句柄（不消耗硬件资源、恒成功），真正的物理资源分配在 `HcommCcuKernelRegister` 阶段（kernel函数执行完后）完成。所有C++包装类均遵循构造即虚拟分配、析构不释放的语义；物理资源随CCU实例生命周期统一管理，不由C++析构释放。

按资源类型分为以下几类：

| 资源类型 | 对应硬件 | 单元分配 | 批量分配 | 通道引用 |
| --- | --- | --- | --- | --- |
| 标量寄存器 | XN | [Variable](Variable.md) | [Array\<Variable\>](Array.md) | [GetResByChannel](GetResByChannel.md) |
| 地址寄存器 | GSA | [Address](Address.md) | — | — |
| 完成事件位 | CKE | [Event](Event.md) | [Array\<Event\>](Array.md) | — |
| 片上MS切片 | MS（4KB） | [CcuBuffer](CcuBuffer.md) | [Array\<CcuBuffer\>](Array.md) | — |
| 本端HBM复合地址 | GSA + XN | [LocalAddr](LocalAddr.md) | — | — |
| 对端HBM复合地址 | GSA + XN | [RemoteAddr](RemoteAddr.md) | — | — |

[Variable](Variable.md)和[Address](Address.md)除资源分配外，还提供赋值与算术运算符；这些运算符描述的是device端执行的操作，在硬件执行时操作对应的寄存器，而非host端立即计算。

## 接口列表

- [Variable](Variable.md)
- [Address](Address.md)
- [Event](Event.md)
- [CcuBuffer](CcuBuffer.md)
- [LocalAddr](LocalAddr.md)
- [RemoteAddr](RemoteAddr.md)
- [Array](Array.md)
- [GetResByChannel](GetResByChannel.md)
