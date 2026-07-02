# CommProtocol

## 功能说明

定义通信协议类型枚举。

## 定义原型

```c
typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  /* 保留协议类型 */
    COMM_PROTOCOL_HCCS = 0,       /* HCCS协议 */
    COMM_PROTOCOL_ROCE = 1,       /* RDMA over Converged Ethernet */
    COMM_PROTOCOL_PCIE = 2,       /* PCIe协议 */
    COMM_PROTOCOL_SIO = 3,        /* SIO协议 */
    COMM_PROTOCOL_UBC_CTP = 4,    /* 华为统一总线UBC_CTP */
    COMM_PROTOCOL_UBC_TP = 5,     /* 华为统一总线UBC_TP */
    COMM_PROTOCOL_UB_MEM = 6,     /* UB_MEM协议 */
    COMM_PROTOCOL_UBOE = 7,       /* UBoE协议 */
    COMM_PROTOCOL_HCCS_ONLY = 8,  /* 一卡双DIE使用HCCS */
} CommProtocol;
```

## 产品支持情况

针对Ascend 950PR/Ascend 950DT，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_UB_MEM
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_CCU
    - COMM_PROTOCOL_UBC_CTP
    - COMM_PROTOCOL_UBC_TP
    
针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
  
针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，各通信引擎支持的通信协议如下：

  - COMM_ENGINE_CPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
