# HcommChannelGetStatus

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

该接口用于查询通信通道的状态，支持批量查询多个通道的连接状态，用于在通道创建后、通信操作前确认通道是否已就绪。

当前为非阻塞式建链，接口直接返回成功并将status置为0，调用方无需持续轮询。

该接口主要用于通道连接建立阶段的同步等待。

## 函数原型

```c
HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| channelList | 输入 | 待查询状态的通道句柄数组，每个元素标识一个已创建的通信通道。<br>ChannelHandle类型的定义请参见[ChannelHandle](../../datatype_definition/ChannelHandle.md)。<br>该参数不能为空指针，数组中每个通道句柄必须是通过[HcommChannelCreate](HcommChannelCreate.md)创建的有效句柄。 |
| listNum | 输入 | 待查询的通道数量。<br>单位为“个”，取值范围：[1, 1048576]。<br>该参数必须大于0。 |
| statusList | 输出 | 通道状态数组，用于返回每个通道的当前状态，与channelList一一对应。<br>该参数不能为空指针。<br>调用者分配的数组，至少包含listNum个元素的空间。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

- channelList数组长度需要与listNum参数一致。
- statusList\[i\]与channelList\[i\]一一对应，表示第i个通道的状态。

## 调用示例

```c
uint32_t channelNum = 50;
std::vector<ChannelHandle> channels(channelNum);
// 参考HcommChannelCreate进行Channel创建
...

uint32_t timeOut = 0;
while (timeOut < 1000) {    
    CHK_RET(HcommChannelGetStatus(channels, channelNum, statuses));    
    result = 0;    
    for (int i = 0; i < channelNum; i++) {       
        result |= statuses[i];    
    }    
    if (result == 0) {break;}
    timeOut += 10;
}

 // 通道就绪，执行通信操作
 // ...
```
