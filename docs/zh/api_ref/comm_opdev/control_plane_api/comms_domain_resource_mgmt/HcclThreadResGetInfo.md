# HcclThreadResGetInfo

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

该接口用于获取Thread底层资源，例如stream等。

## 函数原型

```c
HcclResult HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| thread | 输入 | 线程句柄。<br>ThreadHandle类型的定义可参见[ThreadHandle](../../datatype_definition/ThreadHandle.md)。 |
| resType | 输入 | 底层资源类型（如STREAM等）。<br>ThreadResType类型的定义可参见[ThreadResType](../../datatype_definition/ThreadResType.md)。 |
| infoLen | 输入 | 目标资源信息大小。 |
| info | 输出 | 资源信息输出缓冲区。<br>返回类型为获取的对应资源类型，目前已有资源类型定义如下：<br>typedef aclrtStream ThreadResTypeStream; //stream资源 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

无

## 调用示例

```c
HcclComm comm;
ThreadHandle thread;          //HcclThreadAcquire创建出来的thread的句柄
ThreadResTypeStream stream;   //info缓冲区必须按资源类型对齐且可写
uint32_t size = sizeof(ThreadResTypeStream);  // 必须等于目标类型大小  
CHK_RET(HcclThreadResGetInfo(comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, size, &stream));
//使用stream资源
```
