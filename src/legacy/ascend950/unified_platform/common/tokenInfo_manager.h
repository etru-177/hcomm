/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOKENINFO_MANAGER_H
#define HCCLV2_TOKENINFO_MANAGER_H

#include <mutex>
#include <vector>
#include <unordered_map>
#include <hccl/hccl_types.h>
#include "buffer_key.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

using TokenInfo = std::pair<TokenIdHandle, uint32_t>;
using BufKeyVecIndex = u32;

class TokenInfoManager
{
public:
    TokenInfoManager(u32 devId, RdmaHandle rdmahandle) : devId_(devId), rdmahandle_(rdmahandle)
    {
    }
    
    TokenInfo GetTokenInfo(const BufferKey<uintptr_t, u64> &bufKey);

    void Destroy();

private:
    u32        devId_;
    RdmaHandle rdmahandle_;
    std::mutex tokenInfoMgrMutex_;

    std::unordered_map<BufKeyVecIndex, TokenInfo> tokenInfoMap_;
    std::unordered_map<u32, vector<vector<BufferKey<uintptr_t, u64>>>> bufferKeysMap_; // <devId, BufKeyVecIndex, vector<BufferKey>>

    BufKeyVecIndex GetBufferVecIndex(const BufferKey<uintptr_t, u64> &inputBufKey);
};

bool HasIntersect(const vector<BufferKey<uintptr_t, u64>> &bufKeys, const BufferKey<uintptr_t, u64> &inputBufKey);

} // namespace Hccl

#endif // HCCLV2_TOKENINFO_MANAGER_H