# HcclOpDesc

## 功能说明

用于描述AICPU运行核函数时的算子信息，包含算子类型、名称和通信相关参数。该结构体用于自定义通信算子场景，支持Send和Receive通信任务的描述。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;
```

## 参数说明

- **header**：ABI头部，包含版本等信息。类型定义请参见[CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md) 。
- **opDescType**：算子描述类型。
- **opName**：算子名称，最大长度为256字节（HCCL_OP_DESC_OP_NAME_MAX_LEN）。
- **raws**：原始数据，用于通用场景，长度76字节。
- **p2p**：Send和Receive通信任务的描述参数，[HcclOpP2pDesc](#hcclopp2pdesc)类型，作为联合体成员与raws共用76字节空间。

## HcclOpP2pDesc

用于描述Send和Receive通信任务的详细信息，包含数据缓冲区地址、通信命令类型、数据类型、数据个数、对端rank编号以及展开流等参数。该结构体作为HcclOpDesc的联合体成员使用。

### 定义原型

```c
typedef struct {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;
```

### 参数说明

- **buffer**：数据缓冲区地址，用于发送或接收数据的内存地址。需确保内存已正确分配且可访问。
- **reserved**：预留字段，长度8字节，用于未来扩展。
- **cmdType**：通信命令类型，指定通信操作类型（如HCCL_CMD_SEND、HCCL_CMD_RECEIVE等）。
- **dataType**：数据类型，类型定义请参见[HcclDataType](./HcclDataType.md)。
- **count**：数据个数，指定需要传输的数据元素数量。需与dataType匹配buffer的实际数据大小和类型。
- **remoteRank**：对端rank编号，定通信的对端节点编号。需在通信域的有效rank编号范围内。
- **unfoldStream**：展开流，用于AICPU通信任务的流控制。

## 相关常量

```c
const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;  // 算子名称最大长度
```
