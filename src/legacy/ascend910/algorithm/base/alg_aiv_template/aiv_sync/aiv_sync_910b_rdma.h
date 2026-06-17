/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aiv_communication_base.h"
 
using namespace AscendC;
 
class AivSync910BRdma : public AivCommBase {
public:
    __aicore__ inline AivSync910BRdma() {}
    __aicore__ inline void Process(int32_t tag, uint64_t rmaInfo, int32_t serverNum);
};
 
__aicore__ inline void AivSync910BRdma::Process(int32_t tag, uint64_t rmaInfo, int32_t serverNum)
{
    uint32_t rankPerSever = rankSize_ / serverNum;
    uint64_t flagOffset = ((tag % AIV_PING_PONG_FACTOR_TWO == 0) ? 0 : 2 * rankSize_ * FLAG_SIZE);
    if (blockIdx_ < serverNum) {
        for(uint32_t i = 0; i < rankPerSever; i++) {
            uint32_t targetRank = blockIdx_ + i * serverNum;
            if (targetRank == rank_) {
                continue;
            }
            uint64_t localFlagOffset = 2 * targetRank * FLAG_SIZE;
            __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_IN_RDMA[rank_] + flagOffset + localFlagOffset + FLAG_SIZE); // flag标志位
            SetSignalValue(ctrlFlagGM, localSetTensor, 1);
            PipeBarrier<PIPE_ALL>();
            AIVRDMAPostSend((GM_ADDR)((uint64_t)ctrlFlagGM), (GM_ADDR)((uint64_t)(GM_IN_RDMA[targetRank] + flagOffset + 2 * rank_ * FLAG_SIZE)),
                targetRank, UB_FLAG_PAD_COUNT, (__gm__ HcclRMAInfo*)rmaInfo, false, false);
            PipeBarrier<PIPE_ALL>();
        }

        for (uint32_t i = 0; i < rankPerSever; i++) {
            uint32_t sourceRank = blockIdx_ + i * serverNum;
            if (sourceRank == rank_) {
                continue;
            }
            WaitSignalValue((__gm__ int32_t *)(GM_IN_RDMA[rank_] + flagOffset + 2 * sourceRank * FLAG_SIZE), localCheckTensor, 1);
            PipeBarrier<PIPE_ALL>();
            SetSignalValue((__gm__ int32_t *)(GM_IN_RDMA[rank_] + flagOffset + 2 * sourceRank * FLAG_SIZE), localSetTensor, 0);
        }
    }
}
__aicore__ inline void aiv_sync_910b_rdma(KERNEL_ARGS_DEF)
{
    AivSync910BRdma op;
    op.InitForRDMA(KERNEL_CLASS_INIT, false);
    op.Process(tag, rmaInfo, serverNum);
}
