/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PRIMITIVES_H
#define CCU_PRIMITIVES_H

#include "ccu_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    CCU_DEFAULT = 0,   /**< default, now is scheduler mode */
    CCU_SCHED   = 1,   /**< scheduler mode, with less loop resource */
    CCU_MS      = 2,   /**< memory slices mode, with more loop resouce */
    CCU_UNUSED  = 254,
    CCU_RESERVED = 255 /**< reserved */
} CcuInstanceType;

// 以下接口当前平台层实现预埋，后续对外开放
// 该数据结构当前内部使用，未对外
struct CcuResDesc {
    uint32_t dieId;
    CcuInstanceType insType;
};

//todo: 注释待修改
/**
 * @brief 启用CCU特性，初始化CCU平台层
 *
 * @param deviceLogicId 设备逻辑ID
 * @param ccuDrvHandle CCU驱动句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
extern CcuResult HcommCcuInsCreate(const void *resDesc, uint32_t descNum, CcuInsHandle *insHandle);

/**
 * @brief 关闭CCU特性，解初始化CCU平台层
 *
 * @param deviceLogicId 设备逻辑ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
extern CcuResult HcommCcuInsDestroy(CcuInsHandle insHandle, int32_t curDeviceLogicId);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_PRIMITIVES_H