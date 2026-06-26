/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_kernel_entrance.h"
#include "log.h"
#include "acl/acl_rt.h"

using namespace Hccl;
using u64 = unsigned long long;
std::unordered_map<std::string, std::unordered_map<int32_t, std::unique_ptr<Hccl::TaskService>>> g_taskServiceMap;
extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args)
{
    struct DpuKernelLaunchParam {
        u64      memorySize;
        void*       deviceMem;
        void*       hostMem;
        int32_t    deviceId;
        std::string commId;
    };

    HCCL_INFO("[%s] Launch Dpu Kernel: 0x%lx", __func__, args);
    if (args == 0) {
        HCCL_ERROR("[%s] args is null.", __func__);
        return HCCL_E_PARA;
    }

    if (reinterpret_cast<uint64_t>(args) + sizeof(DpuKernelLaunchParam) < reinterpret_cast<uint64_t>(args)) {
        HCCL_ERROR("[%s] Invalid args address.", __func__);
        return HCCL_E_PARA;
    }
    // 解析参数信息
    DpuKernelLaunchParam *params = reinterpret_cast<DpuKernelLaunchParam *>(args);

    HCCL_RUN_INFO("[%s] DpuKernelLaunchParam{commId:%s; memorySize:%lu; deviceMem:%p; hostMem:%p; devId:%d}", __func__,
                  params->commId.c_str(), params->memorySize, params->deviceMem, params->hostMem, params->deviceId);

    if (params->memorySize == 0) {
        HCCL_ERROR("[%s] memorySize is 0.", __func__);
        return HCCL_E_PARA;
    }
    if (params->deviceMem == nullptr || params->hostMem == nullptr) {
        HCCL_ERROR("[%s] deviceMem[%p] or hostMem[%p] is nullptr.", __func__, params->deviceMem, params->hostMem);
        return HCCL_E_PARA;
    }

    // 实例化TaskService
    std::unique_ptr<Hccl::TaskService> taskService = std::make_unique<Hccl::TaskService>(params->deviceMem, params->memorySize,
                            params->hostMem, params->memorySize);

    aclError ret = aclrtSetDevice(params->deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] set device fail. DeviceId: %d.", __func__, params->deviceId);
        return HCCL_E_RUNTIME;
    }

    // 设置到通信域中保存 map<commId, TaskService>
    HCCL_INFO("[%s] save TaskService", __func__);
    g_taskServiceMap[params->commId][params->deviceId] = std::move(taskService);

    // Run
    HCCL_INFO("[%s] start to TaskRun", __func__);
    g_taskServiceMap.at(params->commId).at(params->deviceId)->TaskRun();
    return HCCL_SUCCESS;
}
}