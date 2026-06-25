# CommEngine

## 功能说明

通信引擎类型枚举。

## 定义原型

```c
typedef enum {
    COMM_ENGINE_RESERVED = -1,    //< 保留的通信引擎
    COMM_ENGINE_CPU = 0,          //< HOST CPU引擎
    COMM_ENGINE_CPU_TS = 1,       //< HOST CPU TS引擎
    COMM_ENGINE_AICPU = 2,        //< AICPU引擎
    COMM_ENGINE_AICPU_TS = 3,     //< AICPU TS引擎
    COMM_ENGINE_AIV = 4,          //< AIV引擎
    COMM_ENGINE_CCU = 5,          //< CCU引擎
} CommEngine;
```
