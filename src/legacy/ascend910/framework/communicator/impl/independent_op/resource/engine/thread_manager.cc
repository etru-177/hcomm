/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_manager.h"
#include <cstring>
#include "aicpu_launch_manager.h"
#include "aicpu_operator_pub.h"
#include "independent_op.h"

namespace hccl {


ThreadMgr::ThreadMgr(uint32_t threadNum, uint32_t notifyNumPerThread, std::string commId, 
    aclrtBinHandle binHandle, const ManagerCallbacks& callbacks) : threadNum_(threadNum), notifyNumPerThread_(notifyNumPerThread), 
    commId_(commId), binHandle_(binHandle), callbacks_(callbacks){}

uint64_t ThreadMgr::GetMaxNotifyTotal()
{
    // 如果没设定最大值，设置一下
    uint64_t maxNotifyTotal = 0;
    if (threadNum_ == HCCL_COMM_THREADNUM_CONFIG_NOT_SET &&
        notifyNumPerThread_ == HCCL_COMM_NOTIFY_NUM_PER_THREAD_CONFIG_NOT_SET) {
        maxNotifyTotal = HCCL_THREAD_NOTIFY_MAX_NUM;
        threadNum_ = SIGNAL_DEV_STREAM_MAX_NUM;
        notifyNumPerThread_ = HCCL_THREAD_NOTIFY_MAX_NUM;
    } else {
        maxNotifyTotal = static_cast<uint64_t>(threadNum_) * static_cast<uint64_t>(notifyNumPerThread_);
        maxNotifyTotal = maxNotifyTotal > HCCL_THREAD_NOTIFY_MAX_NUM ? HCCL_THREAD_NOTIFY_MAX_NUM : maxNotifyTotal;
    }
    return maxNotifyTotal;
}

HcclResult ThreadMgr::CheckNotifyNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
{
    uint64_t maxNotifyTotal = GetMaxNotifyTotal();
    const uint64_t used = usedNotifyNum_;
    uint64_t remainNotifyQuota = (maxNotifyTotal > used) ? (maxNotifyTotal - used) : 0;
    uint64_t needNotifyTotal = static_cast<uint64_t>(threadNum) * static_cast<uint64_t>(notifyNumPerThread);
    if (remainNotifyQuota < needNotifyTotal  || notifyNumPerThread > notifyNumPerThread_ ||
        maxNotifyTotal > HCCL_THREAD_NOTIFY_MAX_NUM) {
        HCCL_ERROR("[ThreadMgr][%s] Notify quota exhausted: remainQuota[%llu], total[%llu], used[%llu], need[%llu], " 
            "setPreNum[%u], allocPreNum[%u]", __func__, remainNotifyQuota, maxNotifyTotal, used, needNotifyTotal,
            notifyNumPerThread_, notifyNumPerThread);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire quota: engine[%d], "
        "remainNotifyQuota[%u]", __func__, commId_.c_str(), engine, remainNotifyQuota);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::CheckThreadNum(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
{
    GetMaxNotifyTotal();
    uint32_t remainQuota = (threadNum_ > threads_.size()) ? (threadNum_ - threads_.size()) : 0;
    if (remainQuota == 0 || threadNum > remainQuota) {
        HCCL_ERROR("[ThreadMgr][%s] Threads quota exhausted: remainQuota[%u], need[%u].",
            __func__, remainQuota, threadNum);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire quota: engine[%d] threadNum[%llu].",
        __func__, commId_.c_str(), engine, remainQuota);
    return CheckNotifyNum(engine, threadNum, notifyNumPerThread);
}

HcclResult ThreadMgr::SupplementNotify(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
{
    NotifyLoadType notifyLoadType = NotifyLoadType::HOST_NOTIFY;
    CHK_RET(CommEngineToNotifyLoadType(engine, notifyLoadType));
    std::vector<std::shared_ptr<Thread>> &threads = engineToThreadsMap_[engine];
    std::vector<std::shared_ptr<Thread>> needSupplementThread;
    HcclResult ret = HCCL_E_INTERNAL;
    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] threadNum[%zu] supplementNotifyNum[%u] notifyLoadType[%u]",
        __func__, commId_.c_str(), threads.size(), notifyNumPerThread, static_cast<int32_t>(notifyLoadType));

    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    u32 aicpuHostNotify = (devType == DevType::DEV_TYPE_950 && (engine == COMM_ENGINE_AICPU_TS ||
        engine == COMM_ENGINE_AICPU)) ? 1 : 0;

    for (uint32_t i = 0; i < threads.size() && i < threadNum; ++i) {
        HCCL_INFO("[ThreadMgr][%s] Hcom[%s] AicpuTsThread notifyLoadType[%u]",
                __func__, commId_.c_str(), static_cast<int32_t>(notifyLoadType));
        u32 notifyNum = threads[i]->GetNotifyNum() >= aicpuHostNotify ?
            (threads[i]->GetNotifyNum() - aicpuHostNotify) : 0;
        if (notifyNumPerThread > notifyNum) {
            u32 supplementNotifyNum = notifyNumPerThread - notifyNum;
            CHK_RET(CheckNotifyNum(engine, 1, supplementNotifyNum));
            ret = threads[i]->SupplementNotify(supplementNotifyNum);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[ThreadMgr][SupplementNotify] Failed to supplement notify[%u] on thread index %u",
                    supplementNotifyNum, i);
                return ret;
            }
            needSupplementThread.push_back(threads[i]);
            usedNotifyNum_ += supplementNotifyNum;
        }
    }

    // thread资源 AICPU侧展开
    if ((engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) && needSupplementThread.size() > 0) {
        if (!callbacks_.getAicpuCommState()) {
            HCCL_INFO("ThreadMgr::SupplementNotify kernelLaunchAicpuCommInit start");
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(ret != HCCL_SUCCESS, 
                HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
            callbacks_.setAicpuCommState(true);
        }

        std::unique_ptr<ThreadHandle[]> threadHandle;
        EXCEPTION_CATCH(threadHandle = std::make_unique<ThreadHandle[]>(needSupplementThread.size()),
            return HCCL_E_PTR);
        for (uint32_t i = 0; i < needSupplementThread.size(); ++i) {
            threadHandle[i] = hostToDeviceThreadHandle_[reinterpret_cast<ThreadHandle>(needSupplementThread[i].get())];
        }
        HCCL_INFO("ThreadMgr::SupplementNotify ThreadKernelLaunchForComm start needSupplementThread size[%zu]",
            needSupplementThread.size());
        ret = AicpuLaunchMgr::SupplementNotifyKernelLaunch(needSupplementThread, commId_, threadHandle, binHandle_);
        HCCL_INFO("ThreadMgr::SupplementNotify ThreadKernelLaunchForComm end");
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][SupplementNotify] AiCpuKernelLaunch failed, return [%d].", ret), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::SupplementThread(CommEngine engine, uint32_t supplementThreadNum, uint32_t notifyNumPerThread)
{
    CHK_RET(CheckThreadNum(engine, supplementThreadNum, notifyNumPerThread));
    NotifyLoadType notifyLoadType;
    StreamType streamType;
    CHK_RET(CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(CommEngineToStreamType(engine, streamType));
    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] supplementThreadNum[%u] notifyNum[%u] notifyLoadType[%u], streamType[%u]",
        __func__, commId_.c_str(), supplementThreadNum, notifyNumPerThread,
        static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType));
    std::vector<std::shared_ptr<Thread>> newThreads;
    newThreads.reserve(supplementThreadNum);
    HcclResult ret = HCCL_E_INTERNAL;

    for (uint32_t i = 0; i < supplementThreadNum; ++i) {
        std::shared_ptr<Thread> handle;
        HCCL_INFO("[ThreadMgr][%s] Hcom[%s] notifyLoadType[%u], streamType[%u]",
                __func__, commId_.c_str(), static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType));
        CHK_RET(CreateThread(engine, streamType, notifyNumPerThread, notifyLoadType, handle));
        ret = handle->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] Failed to init thread index %u", i);
            return ret;
        }
        usedNotifyNum_ += notifyNumPerThread;
        newThreads.emplace_back(std::move(handle));
    }

    // thread资源 AICPU侧展开
    std::unique_ptr<ThreadHandle[]> hostHandle;
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        if (!callbacks_.getAicpuCommState()) {
            HCCL_INFO("ThreadMgr::HcclAllocThreadRes kernelLaunchAicpuCommInit start");
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
            callbacks_.setAicpuCommState(true);
        }

        EXCEPTION_CATCH(hostHandle = std::make_unique<ThreadHandle[]>(newThreads.size()),
            return HCCL_E_PTR);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunchForComm start");
        uint64_t beginTime = Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        ret = AicpuLaunchMgr::ThreadKernelLaunchForComm(newThreads, commId_, hostHandle, binHandle_);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunchForComm end");
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] AiCpuKernelLaunch failed, return [%d].", ret), ret);
        if (callbacks_.reportProfilingKernel != nullptr) {
            ret = callbacks_.reportProfilingKernel(beginTime, "RunAicpuIndOpThreadInit");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] ReportProfilingAiCpuKernelLaunch failed, return [%d].", ret), ret);
        }
    }
    threads_.reserve(threads_.size() + newThreads.size());
    engineToThreadsMap_[engine].reserve(engineToThreadsMap_[engine].size() + newThreads.size());
    engineToThreadsMap_[engine].insert(engineToThreadsMap_[engine].end(), newThreads.begin(), newThreads.end());
    threads_.insert(threads_.end(), newThreads.begin(), newThreads.end());

    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        for (u32 i = 0; i < newThreads.size(); ++i) {
            ThreadHandle cpuTsHandle = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            newThreads[i]->AddThreadHandleToMap(engine, hostHandle[i]);
            hostToDeviceThreadHandle_[cpuTsHandle] = hostHandle[i];
            threadHandleOthersToCpu_[hostHandle[i]] = cpuTsHandle;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquireV2(CommEngine engine, uint32_t threadNum,
    uint32_t notifyNumPerThread, ThreadHandle *threads, std::vector<uint32_t> &threadId)
{
    CHK_PTR_NULL(threads);
    if (threadNum == 0) {
        HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] threadNum is 0");
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(threadMutex_);
    std::lock_guard<std::mutex> lockMap(threadMapMutex_);
    std::lock_guard<std::mutex> engineToThreadMtx(engineToThreadMutex_);
    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire begin, max: engine[%d] threadNum[%u],"
        "notifyPerThread[%u], need: threadNum[%u], notifyPerThread[%u]",
        __func__, commId_.c_str(), engine, threadNum_, notifyNumPerThread_, threadNum, notifyNumPerThread);

    // 1、thread上的notify数量不够，需要给thread补充notify
    auto iter = engineToThreadsMap_.find(engine);
    if (iter == engineToThreadsMap_.end()) {
        engineToThreadsMap_.emplace(engine, std::vector<std::shared_ptr<Thread>>{});
    }
    auto &threadVec = engineToThreadsMap_[engine];
    if (threadVec.size() > 0) {
        // 调用补充函数，如果engine是COMM_ENGINE_AICPU_TS、COMM_ENGINE_AICPU，需要去device恢复
        CHK_RET(SupplementNotify(engine, threadNum, notifyNumPerThread));
    }

    // 2、补充thread
    if (threadVec.size() < threadNum) {
        u32 supplementThreadNum = threadNum - threadVec.size();
        // 调用补充函数，如果engine是COMM_ENGINE_AICPU_TS、COMM_ENGINE_AICPU，需要去device恢复
        CHK_RET(SupplementThread(engine, supplementThreadNum, notifyNumPerThread));
    }
    // 3、返回threadHandle和id
    for (u32 idx = 0; idx < threadNum; idx++) {
        ThreadHandle handle = reinterpret_cast<ThreadHandle>(threadVec[idx].get());
        threads[idx] = (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU) ?
            hostToDeviceThreadHandle_[handle] : handle;
        uint32_t id = threadVec[idx]->GetStream()->sqId();
        HCCL_DEBUG("[%s]idx[%u] threadHandle[%llu] thread id = [%u]", __func__, idx, threads[idx], id);
        threadId.push_back(id);

        std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
        threadMap_[threads[idx]] = threadVec[idx];
    }

    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire done: engine[%d] threadNum[%u],"
        "notifyPerThread[%u]%s", __func__, commId_.c_str(), engine, threadNum, notifyNumPerThread,
        (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) ? " (AICPU token ready)" : "");
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquire(CommEngine engine, uint32_t threadNum,
    uint32_t notifyNumPerThread, ThreadHandle *threads, std::vector<uint32_t> &threadId)
{
    CHK_PTR_NULL(threads);
    std::lock_guard<std::mutex> lock(threadMutex_);
    std::lock_guard<std::mutex> lockMap(threadMapMutex_);
    HCCL_INFO("[ThreadMgr][%s] Hcom[%s] HcclThreadAcquire begin, max: engine[%d] threadNum[%u],"
        "notifyPerThread[%u], need: threadNum[%u], notifyPerThread[%u]",
        __func__, commId_.c_str(), engine, threadNum_, notifyNumPerThread_, threadNum, notifyNumPerThread);

    if (threadNum == 0) {
        HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] threadNum is 0");
        return HCCL_E_PARA;
    }

    CHK_RET(CheckThreadNum(engine, threadNum, notifyNumPerThread));

    NotifyLoadType notifyLoadType;
    StreamType streamType;
    CHK_RET(CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(CommEngineToStreamType(engine, streamType));
    std::vector<std::shared_ptr<Thread>> newThreads;
    newThreads.reserve(threadNum);
    HcclResult ret = HCCL_E_INTERNAL;

    for (uint32_t i = 0; i < threadNum; ++i) {
        std::shared_ptr<Thread> handle;
        HCCL_INFO("[ThreadMgr][%s] Hcom[%s] AicpuTsThread notifyLoadType[%u], streamType[%u]",
                __func__, commId_.c_str(), static_cast<int32_t>(notifyLoadType), static_cast<int32_t>(streamType));
        CHK_RET(CreateThread(engine, streamType, notifyNumPerThread, notifyLoadType, handle));
        ret = handle->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] Failed to init thread index %u", i);
            return ret;
        }
        usedNotifyNum_ += notifyNumPerThread;
        newThreads.emplace_back(std::move(handle));
    }

    // thread资源 AICPU侧展开
    std::unique_ptr<ThreadHandle[]> hostHandle;
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        if (!callbacks_.getAicpuCommState()) {
            HCCL_INFO("ThreadMgr::HcclAllocThreadRes kernelLaunchAicpuCommInit start");
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(ret != HCCL_SUCCESS, 
                HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
            callbacks_.setAicpuCommState(true);
        }

        EXCEPTION_CATCH(hostHandle = std::make_unique<ThreadHandle[]>(newThreads.size()),
            return HCCL_E_PTR);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunch start");
        ret = AicpuLaunchMgr::ThreadKernelLaunchForComm(newThreads, commId_, hostHandle, binHandle_);
        HCCL_INFO("ThreadMgr::HcclAllocThreadRes ThreadKernelLaunch end");
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ThreadMgr][HcclThreadAcquire] AiCpuKernelLaunch failed, return [%d].", ret), ret);
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = hostHandle[i];
            HCCL_INFO("[ThreadMgr][%s] aicpu threadArray[%u] = [%lu]", __func__, i, threads[i]);
        }
    } else {
        for (size_t i = 0; i < newThreads.size(); ++i) {
            threads[i] = reinterpret_cast<ThreadHandle>(newThreads[i].get());
            HCCL_INFO("[ThreadMgr][%s] host threadArray[%u] = [%lu]", __func__, i, threads[i]);
        }
    }
    for (size_t i = 0; i < newThreads.size(); ++i) {
        uint32_t id = newThreads[i]->GetStream()->id();
        HCCL_DEBUG("[%s] thread id = [%u]", __func__, id);
        threadId.push_back(id);
    }
    threads_.reserve(threads_.size() + newThreads.size());
    auto threadsIt = threads_.end();
    threads_.insert(threads_.end(),
                    std::make_move_iterator(newThreads.begin()),
                    std::make_move_iterator(newThreads.end()));

    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        for (u32 i = 0; i < newThreads.size(); ++i, ++threadsIt) {
            ThreadHandle cpuTsHandle = reinterpret_cast<ThreadHandle>((*threadsIt).get());
            (*threadsIt)->AddThreadHandleToMap(engine, hostHandle[i]);
            threadHandleOthersToCpu_[hostHandle[i]] = cpuTsHandle;
        }
    }

    HCCL_INFO("[ThreadMgr][HcclThreadAcquire] Hcom[%s] HcclThreadAcquire done: engine[%d] threadNum[%u],"
        "notifyPerThread[%u]%s", commId_.c_str(), engine, threadNum, notifyNumPerThread,
        (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) ? " (AICPU token ready)" : "");
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclGetNotifyNumInThread(ThreadHandle thread, uint32_t *notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    Thread* hcclThread = reinterpret_cast<Thread*>(thread);
    CHK_PTR_NULL(hcclThread);
    *notifyNum = hcclThread->GetNotifyNum();
    HCCL_INFO("[ThreadMgr] Hcom[%s] HcclGetNotifyNumInThread done: notifyPerThread[%u]",
        commId_.c_str(),  *notifyNum);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadAcquireWithStream(CommEngine engine,
    rtStream_t stream, uint32_t notifyNum, ThreadHandle *thread)
{
    CHK_PTR_NULL(thread);

    if (mainThread_.find(stream) != mainThread_.end()) {
        if (mainThread_[stream]->GetNotifyNum() < notifyNum) {
            u32 supplementNotifyNum = notifyNum - mainThread_[stream]->GetNotifyNum();
            CHK_RET(mainThread_[stream]->SupplementNotify(supplementNotifyNum));
        }
        *thread = reinterpret_cast<ThreadHandle>(mainThread_[stream].get());
        return HCCL_SUCCESS;
    }

    NotifyLoadType notifyLoadType;
    CHK_RET(CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<CpuTsThread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<CpuTsThread>(stream, notifyNum, notifyLoadType), return HCCL_E_PTR);
    CHK_RET(handle->Init());
    handle->SetIsMaster(true);

    // 返回第一个句柄
    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    mainThread_.emplace(stream, std::move(handle));
    *thread = reinterpret_cast<ThreadHandle>(mainThread_[stream].get());
    std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
    threadMap_[*thread] = mainThread_[stream];
    HCCL_INFO("[ThreadMgr] Hcom[%s] HcclThreadAcquireWithStream done: engine[%d] stream[%p],"
        "notifyNum[%u]", commId_.c_str(), engine, stream, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::ThreadExportToCommEngineCpu(uint32_t threadNum, const ThreadHandle *threads, ThreadHandle *exportedThreads)
{
    std::lock_guard<std::mutex> lock(threadMapMutex_);
    for (u32 i = 0; i < threadNum; i++) {
        if (threadHandleOthersToCpu_.find(threads[i]) == threadHandleOthersToCpu_.end()) {
            HCCL_ERROR("[CommEngineResMgr]%s Unknown ThreadHandle[%lu]", __func__, threads[i]);
            return HCCL_E_PARA;
        }
        exportedThreads[i] = threadHandleOthersToCpu_[threads[i]];
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::GetExportedThread(const ThreadHandle threadHandle, CommEngine commEngine, Thread *&exportedThread, std::shared_ptr<Thread> &threadOut)
{
    Thread *threadPtr = reinterpret_cast<Thread *>(threadHandle);
    for (auto &thread : threads_) {
        if (thread.get() == threadPtr) {
            exportedThread = thread->FindThreadByCommEngine(commEngine);
            threadOut = thread;
            return HCCL_SUCCESS;
        }
    }

    for (auto &pair : mainThread_) {
        if (pair.second.get() == threadPtr) {
            exportedThread = pair.second->FindThreadByCommEngine(commEngine);
            threadOut = pair.second;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[ThreadMgr][%s]Unknown ThreadHandle[%lu]", __func__, threadHandle);
    return HCCL_E_PARA;
}

HcclResult ThreadMgr::ThreadExportToCommEngineAicpu(uint32_t threadNum, const ThreadHandle *threads, CommEngine dstCommEngine, ThreadHandle *exportedThreads)
{
    std::vector<std::shared_ptr<Thread>> hostThreads;
    std::vector<u32> index;
    Thread *exportedThread;
    for (u32 i = 0; i < threadNum; i++) {
        std::shared_ptr<Thread> handle;
        CHK_RET(GetExportedThread(threads[i], dstCommEngine, exportedThread, handle));
        if (exportedThread != nullptr) {
            exportedThreads[i] = reinterpret_cast<ThreadHandle>(exportedThread);
            continue;
        } else {
            hostThreads.push_back(handle);
            index.push_back(i);
        }
    }
    if (!hostThreads.empty()) {
        std::lock_guard<std::mutex> lock(threadMapMutex_);
        if (!callbacks_.getAicpuCommState()) {
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
            callbacks_.setAicpuCommState(true);
        }
        std::unique_ptr<ThreadHandle[]> aicpuHandle;
        EXCEPTION_CATCH(aicpuHandle = std::make_unique<ThreadHandle[]>(hostThreads.size()),
                         return HCCL_E_PTR);
        uint64_t beginTime = Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchForComm(hostThreads, commId_, aicpuHandle, binHandle_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[ThreadMgr][HcclThreadExportToCommEngine] AiCpuKernelLaunch failed, return [%d].", ret), ret);
        if (callbacks_.reportProfilingKernel != nullptr) {
            ret = callbacks_.reportProfilingKernel(beginTime, "RunAicpuIndOpThreadInit");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ThreadMgr][HcclThreadExportToCommEngine] ReportProfilingAiCpuKernelLaunch failed, return [%d].", ret), ret);
        }
        for (size_t i = 0; i < hostThreads.size(); ++i) {
            exportedThreads[index[i]] = aicpuHandle[i];
            CHK_RET(hostThreads[i]->AddThreadHandleToMap(dstCommEngine, aicpuHandle[i]));
            threadHandleOthersToCpu_[aicpuHandle[i]] = threads[index[i]];
            HCCL_INFO("[ThreadMgr][%s] aicpu threadArray[%u] = [%lu]", __func__, i, aicpuHandle[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadExportToCommEngine(uint32_t threadNum, const ThreadHandle *threads, CommEngine dstCommEngine, ThreadHandle *exportedThreads)
{
    switch (dstCommEngine) {
    case COMM_ENGINE_CPU_TS:
    case COMM_ENGINE_CPU:
    case COMM_ENGINE_CCU:
        CHK_RET(ThreadExportToCommEngineCpu(threadNum, threads, exportedThreads));
        break;
    case COMM_ENGINE_AICPU:
    case COMM_ENGINE_AICPU_TS:
        CHK_RET(ThreadExportToCommEngineAicpu(threadNum, threads, dstCommEngine, exportedThreads));
        break;
    case COMM_ENGINE_AIV:
    default:
        HCCL_ERROR("[ThreadMgr] Unknown comm engine type: %d", dstCommEngine);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadMgr::HcclThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
{
    CHK_PRT_RET(resType != ThreadResType::THREAD_RES_TYPE_STREAM, HCCL_ERROR("[%s] failed. resType[%d] is not supported.", 
        __func__, static_cast<int32_t>(resType)), HCCL_E_NOT_SUPPORT);

    std::lock_guard<std::mutex> threadhandleToThreadMtx(threadhandleToThreadMutex_);
    auto it = threadMap_.find(thread);
    CHK_PRT_RET(it == threadMap_.end(), 
        HCCL_ERROR("[%s] failed to find handle mapping in threadMap_, thread[0x%llx].", __func__, thread), HCCL_E_NOT_FOUND);
    std::shared_ptr<Thread> threadPtr = it->second;
    CHK_PTR_NULL(threadPtr);
    if (resType == ThreadResType::THREAD_RES_TYPE_STREAM) {
        CHK_PRT_RET(infoLen != sizeof(ThreadResTypeStream), HCCL_ERROR("[%s] failed. infoLen[%u] is mismatch sizeof(ThreadResTypeStream)[%zu]", 
                    __func__, infoLen, sizeof(ThreadResTypeStream)), HCCL_E_PARA);
        CHK_PTR_NULL(threadPtr->GetStream());
        ThreadResTypeStream stream = threadPtr->GetStream()->ptr();
        CHK_PTR_NULL(stream);
        *info = stream;
    } else {
        HCCL_ERROR("[%s] unsupported resType[%d]", __func__, static_cast<int32_t>(resType));
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[%s] success. thread[0x%llx] resType[%d] info[%p]", __func__, thread, static_cast<int32_t>(resType), *info);
    return HCCL_SUCCESS;
}

}
