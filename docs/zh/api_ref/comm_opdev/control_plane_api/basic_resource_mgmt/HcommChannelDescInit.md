# HcommChannelDescInit

## 产品支持情况

- Atlas 350 加速卡：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

初始化通信通道描述列表。

## 函数原型

```c
HcommResult HcommChannelDescInit(HcommChannelDesc *channelDesc, uint32_t descNum)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channelDesc | 输入/输出 | 通信通道描述列表，列表长度为descNum，函数会初始化该结构体。<br>HcommChannelDesc类型的定义可参见[HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md)。 |
| descNum | 输入 | 通信通道描述数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- HcommChannelDesc结构体必须调用该接口进行初始化。

## 调用示例

以初始化通信通道描述数量为2的通信通道描述列表为例：

```c
uint32_t channelNum = 2;
std::vector<HcommChannelDesc> channelDesc(channelNum);
HcommChannelDescInit(channelDesc.data(), channelNum);
```
