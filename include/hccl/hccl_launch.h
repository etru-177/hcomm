/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_LAUNCH_H_
#define HCCL_LAUNCH_H_

#include <stdint.h>
#include <hccl/hccl_types.h>
#include <acl/acl_rt.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ARG_SIZE 8192U
typedef struct P2pParamDef
{
    ThreadHandle sendRecvStream;
    uint8_t opParams[MAX_ARG_SIZE];
}P2pParam;

typedef struct HcclOpP2pDescDef {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;

const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;
const uint32_t HCCL_OP_DESC_RESERVED_LEN = 64;

typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t reserved[256];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;

const uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;
const uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;

typedef struct {
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    void *args;
    uint32_t argSize;
} HcclKernelFuncInfo;

/**
 * @brief Launch AICPU kernel for P2P task
 *
 * @param comm Communication domain handle
 * @param opInfo Operator descriptor parameters
 * @param funcInfo Kernel function info (dynamic library name, kernel function name, args and argSize)
 * @param aicpuThreadHandle AICPU communication main stream thread handle
 * @param userStream User host main stream
 * @return HcclResult
 *   HCCL_SUCCESS: success
 *   HCCL_E_NOT_SUPPORT: HcclGroupStart not called
 *   HCCL_E_INTERNAL: internal error (async task failure, communication task failure)
 */

extern HcclResult HcclAicpuKernelLaunch(HcclComm comm, const HcclOpDesc *opInfo, const HcclKernelFuncInfo *funcInfo,
    ThreadHandle aicpuThreadHandle, aclrtStream userStream);

#ifdef __cplusplus
}
#endif

#endif
