# HcclOpExpansionMode

## 功能说明

定义通信算子的展开模式。

## 定义原型

```c
typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode mode;

typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID  = -1,  /* 无效模式，未初始化或保留。 */
    HCCL_OP_EXPANSION_MODE_AI_CPU   = 0,   /* 在Device侧AI CPU上展开。 */
    HCCL_OP_EXPANSION_MODE_AIV      = 1,   /* 在Device侧Vector Core (AIV)上展开。 */
    HCCL_OP_EXPANSION_MODE_HOST     = 2,   /* 在Host侧CPU上展开，Device侧根据硬件型号自动选择调度器。 */
    HCCL_OP_EXPANSION_MODE_HOST_TS  = 3,   /* 在Host侧CPU上展开，由Host向Device Task Scheduler下发任务，Device侧进行调度执行。 */
} HcclOpExpansionMode;
```
