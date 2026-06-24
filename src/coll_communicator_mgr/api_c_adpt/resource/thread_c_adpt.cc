/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include "hccl/hccl_res.h"
#include "hccl_mem.h"
#include "stream_pub.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "hccl_independent_common.h"
#include "coll_comm_profiling.h"
#include "aicpu_operator_pub.h"
using namespace hccl;
constexpr u32 MAX_EXPORT_THREAD_NUM = 40U;

HcclResult HcclGetNotifyNumInThread(HcclComm comm, ThreadHandle thread,
    CommEngine engine, uint32_t *notifyNum)
{
    CHK_PRT_RET(comm == nullptr,  HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(!IsValidCommEngine(engine), 
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);
    CHK_PRT_RET(notifyNum == nullptr,  HCCL_ERROR("[%s] notifyNum is null", __func__), HCCL_E_PTR);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] engine[%u]", __func__, commId.c_str(), engine);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclGetNotifyNumInThread(thread, engine, notifyNum);
    }
    else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclGetNotifyNumInThread(thread, engine, notifyNum);
    }
    
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclGetNotifyNumInThread] Failed to get notifyNum for engine[%d] ret[%d]", engine, ret);
        return ret;
    }
    HCCL_INFO("[HcclGetNotifyNumInThread] threads for engine[%d], notifyNum[%u]", engine, *notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
    uint32_t notifyNumPerThread, ThreadHandle *threads)
{
    CHK_PRT_RET(comm == nullptr,  HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(threads == nullptr,  HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(!IsValidCommEngine(engine), 
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] engine[%u] reqThreadNum[%u] notifyNumPerThread[%u]",
        __func__, commId.c_str(), engine, threadNum, notifyNumPerThread);

    HcclResult ret = HCCL_SUCCESS;
    std::vector<uint32_t> threadId;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadAcquireV2(engine, threadNum, notifyNumPerThread, threads, threadId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] failed to create threads for engine[%d],threadsNum[%u], notifyNumperThread[%u]",
            __func__, engine, threadNum, notifyNumPerThread);
            return ret;
        }
        Mc2CommInfo mc2CommInfo;
        mc2CommInfo.FreeStreamId = 0;
        mc2CommInfo.streamsId = threadId;
        mc2CommInfo.groupname = commId;
        mc2CommInfo.myRankId = collComm->GetMyRankId();
        mc2CommInfo.rankSize = collComm->GetRankSize();
        CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        CHK_PTR_NULL(hcclCommDfx);
        if (engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_AICPU) {
            hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
        } else {
            auto hcclCommDfxCallBack = collComm->GetDfxCallback();
            for (u32 num = 0; num < threadNum; ++num) {
                int hert = HcommThreadRegisterDfx(threads[num], hcclCommDfxCallBack);
                if (hert != HCCL_SUCCESS) {
                    HCCL_ERROR("[HcclThreadAciqure] ReportThreadAciqureKernel HcommThreadRegisterDfx failed hert:[%d],num:[%d]",
                        hert, num);
                    return HCCL_E_PTR;
                }
            }
        }
        return HCCL_SUCCESS;
    }
    else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadAcquire(engine, threadNum, notifyNumPerThread, threads, threadId);
        if (engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_AICPU) {
            // 上报流
            if (threadNum != threadId.size()) {
                HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%u]", __func__, threadNum, threadId.size());
                return HCCL_E_PARA;
            }
            CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));
        }
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to create threads for engine[%d], threadNum[%u], notifyNumPerThread[%u]",
            __func__, engine, threadNum, notifyNumPerThread);
        return ret;
    }

    HCCL_INFO("[%s] Allocated %u threads for engine[%d], notifyPerThread[%u]", __func__,
              threadNum, engine, notifyNumPerThread);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine,
    aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(thread);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO("Entry-%s:comm[%s] engine[%u] notifyNum[%u] stream[%p]",
        __func__, commId.c_str(), engine, notifyNum, stream);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadAcquireWithStream(engine, stream, notifyNum, thread);
        auto hcclCommDfxCallback = collComm->GetDfxCallback();
        int hret = HcommThreadRegisterDfx(*thread, hcclCommDfxCallback);
        if (hret != 0) {
            HCCL_ERROR("[HcclThreadAcquire] HcclThreadAcquire  HcommThreadRegisterDfx failed");
            return HCCL_E_PTR;
        }
        if (engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_AICPU) {
            Thread *threadPtr = reinterpret_cast<Thread *>(*thread);
            CHK_PTR_NULL(threadPtr);
            Stream *threadStream = threadPtr->GetStream();
            CHK_PTR_NULL(threadStream);
            Mc2CommInfo mc2CommInfo;
            mc2CommInfo.FreeStreamId = 0;
            mc2CommInfo.streamsId.push_back(static_cast<u32>(threadStream->sqId()));
            mc2CommInfo.groupname = commId;
            mc2CommInfo.myRankId = collComm->GetMyRankId();
            mc2CommInfo.rankSize = collComm->GetRankSize();
            CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
            HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
            CHK_PTR_NULL(hcclCommDfx);
            hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
        }
    }
    else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadAcquireWithStream(engine, stream, notifyNum, thread);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclThreadAcquireWithStream] Failed to create thread for engine[%d]", engine);
        return ret;
    }

    HCCL_INFO("[HcclThreadAcquireWithStream] Allocated thread for engine[%d], stream[%p], notifyNum[%u]",
              engine, stream, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclAllocNotify(HcclComm comm, CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum,
    NotifyHandle **notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(!IsValidCommEngine(commEngine), 
        HCCL_ERROR("[%s] commEngine[%u] is invalid", __func__, commEngine), HCCL_E_PARA);
    CHK_PRT_RET(!IsValidNotify(notifyType), 
        HCCL_ERROR("[%s] notifyType[%u] is invalid", __func__, notifyType), HCCL_E_PARA);
    CHK_PRT_RET(notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, 
        HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum), HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(*notifyHandleList != nullptr, HCCL_ERROR("[%s] notifyHandleList is not null", __func__), HCCL_E_PARA);

    if (commEngine == CommEngine::COMM_ENGINE_CPU || commEngine == CommEngine::COMM_ENGINE_CPU_TS
        || commEngine == CommEngine::COMM_ENGINE_CCU) {
        if (notifyType != ::NOTIFY_TYPE_RTS_NOTIFY) {
            HCCL_ERROR("[%s] commEngine[%u] and notifyType[%u] are mismatch",  __func__, commEngine, notifyType);
            return HCCL_E_PARA;
        }
    } else {
        if (notifyType != ::NOTIFY_TYPE_DEVICE_MEM) {
            HCCL_ERROR("[%s] commEngine[%u] and notifyType[%u] are mismatch",  __func__, commEngine, notifyType);
            return HCCL_E_PARA;
        }
    }
 
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] commEngine[%u] notifyType[%u] notifyNum[%p]",
        __func__, commId.c_str(), commEngine, notifyType, notifyNum);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    }
    else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to create notify for commEngine[%d]",  __func__, commEngine);
        return ret;
    }
 
    HCCL_RUN_INFO("[%s] Allocated notify for commEngine[%d], notifyType[%p], notifyNum[%u]", __func__,
        commEngine, notifyType, notifyNum);
    return HCCL_SUCCESS;
}
 
HcclResult HcommFreeNotify(HcclComm comm, uint32_t notifyNum, NotifyHandle *notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, 
        HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum), HCCL_E_PARA);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] notifyNum[%u]", __func__, commId.c_str(), notifyNum);
        HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcommFreeNotify(notifyNum, notifyHandleList);
    }
    else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcommFreeNotify(notifyNum, notifyHandleList);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to free notify",  __func__);
        return ret;
    }
 
    HCCL_RUN_INFO("[%s] Free notify for notifyNum[%u]", __func__, notifyNum);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum, const ThreadHandle *threads, CommEngine dstCommEngine, ThreadHandle *exportedThreads)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(exportedThreads);
    CHK_PRT_RET(!IsValidCommEngine(dstCommEngine),
                HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(dstCommEngine)), HCCL_E_PARA);
    if (threadNum == 0 || threadNum > MAX_EXPORT_THREAD_NUM) {
        HCCL_ERROR("[%s] threadNum[%u] is 0 or greater than %u", __func__, threadNum, MAX_EXPORT_THREAD_NUM);
        return HCCL_E_PARA;
    }

    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO("Entry-[%s]:comm[%s], threadNum[%u], commEngine[%d], threadsPtr[%p], exportedThreadsPtr[%p]", 
             __func__, commId.c_str(), threadNum, dstCommEngine, threads, exportedThreads);
    HcclResult ret;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    } else {
        auto &engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Thread export failed. Export threadNum[%u], commEngine[%d], threadsPtr[%p], exportedThreadsPtr[%p]",
         __func__, threadNum, dstCommEngine, threads, exportedThreads), ret);
    HCCL_INFO("[%s]:comm[%s] export success. ", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif

HcclResult HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(info);
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO("Entry-[%s]:comm[%s], thread[0x%llx], resType[%d], infoLen[%u], info[%p]",
             __func__, commId.c_str(), thread, static_cast<int32_t>(resType), infoLen, info);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadResGetInfo(thread, resType, infoLen, info);
    } else {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (devType != DevType::DEV_TYPE_910B) { // 910B HOST网卡需要走此流程，不打印错误日志
            HCCL_ERROR("[%s] communicatorType is not supported.", __func__);
        }
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] thread resource get info failed. thread[0x%llx], resType[%d], infoLen[%u], info[%p]",
         __func__, thread, static_cast<int32_t>(resType), infoLen, info), ret);
    HCCL_INFO("[%s]:comm[%s] get thread resource success. ", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
