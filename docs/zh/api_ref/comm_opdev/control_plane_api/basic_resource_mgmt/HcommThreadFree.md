# HcommThreadFree

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

使用HcommThreadAlloc接口申请的thread需要通过本接口进行释放。

## 函数原型

```c
HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| threads | 输入 | 通信线程句柄<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |
| threadNum | 输入 | 通信线程数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

只能释放HcommThreadAlloc接口申请的thread。

## 调用示例

```c
ThreadHandle thread[2];
HcommResult ret =  HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
HcommResult ret =  HcommThreadFree(thread, 2);
```
