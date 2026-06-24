# HcommChannelGetNotifyNum

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

获取指定channel的Notify数量。

## 函数原型

```c
HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| channelHandle | 输入 | 通信通道句柄。<br>ChannelHandle类型的定义可参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。 |
| notifyNum | 输出 | 指定channel的notify数量。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

支持的通信协议包括：RoCE、UBC_TP、UBC_CTP。

## 调用示例

```c
// AI CPU引擎需要DEVICE类型的端点
 EndpointDesc deviceEp = {0};
 deviceEp.protocol = COMM_PROTOCOL_ROCE;
 deviceEp.commAddr.type = COMM_ADDR_TYPE_IP_V4;
 inet_pton(AF_INET, "192.168.1.10", &deviceEp.commAddr.addr);
 deviceEp.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
 deviceEp.loc.device.devPhyId = 0;
 deviceEp.loc.device.superPodId = 0;

 EndpointHandle endpointHandle = nullptr;
 HcommResult ret = HcommEndpointCreate(&deviceEp, &endpointHandle);
 if (ret != 0) {
     printf("Failed to create device endpoint, ret = %d\n", ret);
     return ret;
 }

 // 准备通道描述符
 const uint32_t CHANNEL_NUM = 3;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};

 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // 填充远端端点信息
     channelDescs[i].remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
     channelDescs[i].remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
     char remoteIp[32] = {0};
     snprintf(remoteIp, sizeof(remoteIp), "192.168.2.%d", i + 1);
     inet_pton(AF_INET, remoteIp, &channelDescs[i].remoteEndpoint.commAddr.addr);
     channelDescs[i].remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
     channelDescs[i].remoteEndpoint.loc.device.devPhyId = i + 1;
     channelDescs[i].notifyNum = 32;
     // RoCE参数
     channelDescs[i].roceAttr.queueNum = 16;
     channelDescs[i].roceAttr.retryCnt = 5;
 }

 // 创建AI CPU引擎的通道
 ChannelHandle channels[CHANNEL_NUM] = {0};
 ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU, channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create AICPU channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }
uint32_t *notifyNum;
ret = HcommChannelGetNotifyNum(channels[0], notifyNum);
```
