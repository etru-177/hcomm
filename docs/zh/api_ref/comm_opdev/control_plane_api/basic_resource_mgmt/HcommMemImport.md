# HcommMemImport

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

本端通过导入对端导出的内存描述，获取相应的内存描述信息。

## 函数原型

```c
HcommResult HcommMemImport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen, CommMem *outMem)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| endpointHandle | 输入 | Endpoint句柄。<br>EndpointHandle类型的定义请参见[EndpointHandle](../../datatype_definition/EndpointHandle.md)。 |
| memDesc | 输入 | 描述信息指针。 |
| descLen | 输入 | 描述信息长度。 |
| outMem | 输出 | 内存段元数据描述符。<br>CommMem类型的定义请参见[CommMem](../../datatype_definition/CommMem.md)。 |

## 返回值

HcommResult：接口成功返回0，其他失败。

## 约束说明

无

## 调用示例

```c
// export端操作
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
const char *memTag = "HcclBuffer";
CommMem mem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = reinterpret_cast<void*>(0x1111),
    .size = 100
};
HcommMemHandle memHandle;
result = HcommMemReg(endpointHandle, memTag, &mem, &memHandle);

uint32_t memDescLen = 0;
void* memDesc = nullptr;
result = HcommMemExport(endpointHandle, memHandle, &memDesc, &memDescLen);

// import端操作
CommMem outMem;
result = HcommMemImport(endpointHandle, memDesc, memDescLen, &outMem);
```
