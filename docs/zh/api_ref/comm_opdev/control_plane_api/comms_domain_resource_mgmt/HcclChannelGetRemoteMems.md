# HcclChannelGetRemoteMems

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

获取通信通道中交换的远端内存信息。

## 函数原型

```c
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t *memNum, CommMem **remoteMems, char ***memTags);
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| comm | 输入 | 通信域句柄。<br>HcclComm类型的定义如下：<br>typedef void *HcclComm; |
| channel | 输入 | 通信通道句柄。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。 |
| memNum | 输出 | 内存数量。 |
| remoteMems | 输出 | 远端内存列表。<br>CommMem类型的定义请参见[CommMem](../../datatype_definition/CommMem.md)。 |
| memTags | 输出 | 远端内存字符串列表。 |

## 返回值

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

仅Ascend 950PR/Ascend 950DT的AIV引擎支持。

## 调用示例

```c
uint32_t channelNum = 1;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);

// 省略channelDesc配置

HcclComm comm;
CommEngine engine = CommEngine::COMM_ENGINE_AIV;
std::vector<ChannelHandle> channels(channelNum);
HcclChannelAcquire(comm, engine, channelDesc.data(), channelNum, channels.data());

uint32_t memNum = 0;
CommMem* remoteMems = nullptr;
char** memTags = nullptr;
HcclChannelGetRemoteMems(comm, channels[0], &memNum, &remoteMems, &memTags);
```
