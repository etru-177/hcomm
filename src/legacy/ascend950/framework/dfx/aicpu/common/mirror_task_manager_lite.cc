/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mirror_task_manager_lite.h"

namespace Hccl {

MirrorTaskManagerLite::MirrorTaskManagerLite()
{
}

void MirrorTaskManagerLite::RegFullyCallBack(std::function<void(const std::string&, u32)> callBack)
{
    fullyNewCallBack_ = callBack;
    return;
}

void MirrorTaskManagerLite::RegFullyCallBack(std::function<void()> callBack)
{
    fullyCallBack_ = callBack;
    return;
}

void MirrorTaskManagerLite::AddTaskInfo(std::shared_ptr<TaskInfo> taskInfo)
{
    HCCL_INFO("[MirrorTaskManagerLite][AddTaskInfo]AddTaskInfo begin");
    if (UNLIKELY(taskInfo == nullptr)) {
        THROW<InternalException>(
            StringFormat("MirrorTaskManagerLite::AddTaskInfo taskInfo is nullptr"));
    }

    if (taskInfo->dfxOpInfo_ == nullptr) {
        taskInfo->dfxOpInfo_ = currDfxOpInfo_;
    }

    if (queueMap_.find(taskInfo->streamId_) == queueMap_.end()) {
        queueMap_[taskInfo->streamId_] = std::make_unique<CircularQueue<std::shared_ptr<TaskInfo>>>(MAX_AICPU_CIRCULAR_QUEUE_LENGTH);
        queueTaskNum[taskInfo->streamId_] = 0;
    }

    if (queueTaskNum[taskInfo->streamId_] == static_cast<u32>(queueMap_[taskInfo->streamId_]->Capacity())) {
        fullyCallBack_();
        queueTaskNum[taskInfo->streamId_] = 0;
    }

    queueMap_[taskInfo->streamId_]->Append(taskInfo);
    queueTaskNum[taskInfo->streamId_]++;

    HCCL_INFO("[MirrorTaskManagerLite][AddTaskInfo]add streamId(sqId)[%u] taskId(sqeId)[%u] queueMapsize[%u]", taskInfo->streamId_, taskInfo->taskId_, queueMap_.size());

    return;
}

bool MirrorTaskManagerLite::IsStaticGraphMode(const CollOperator &collOperator) const
{
    return (collOperator.staticAddr == false) && (collOperator.staticShape == false);
}

void MirrorTaskManagerLite::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo)
{
    if (dfxOpInfo == nullptr) {
        HCCL_ERROR("[MirrorTaskManagerLite][SetCurrDfxOpInfo]fail, dfxOpInfo is nullptr");
        return;
    }
    currDfxOpInfo_     = dfxOpInfo;
    isStaticGraphMode_ = IsStaticGraphMode(dfxOpInfo->op_);
    opMode_            = dfxOpInfo->op_.opMode;
    HCCL_INFO("[MirrorTaskManagerLite][SetCurrDfxOpInfo] Succeed, currDfxOpInfo_[%p], this[%p] !", currDfxOpInfo_.get(), this);
    return;
}

std::shared_ptr<DfxOpInfo> MirrorTaskManagerLite::GetCurrDfxOpInfo() const
{
    HCCL_INFO("[MirrorTaskManagerLite][GetCurrDfxOpInfo] Succeed, currDfxOpInfo_[%p], this[%p] !", currDfxOpInfo_.get(), this);
    return currDfxOpInfo_;
}

TaskInfoQueue *MirrorTaskManagerLite::GetQueue(u32 streamId) const
{
    if (queueMap_.find(streamId) == queueMap_.end()) {
        HCCL_ERROR("MirrorTaskManagerLite::GetQueue streamId(sqId)[%u] out of range", streamId);
        return nullptr;
    }
    return queueMap_.find(streamId)->second.get();
}

std::shared_ptr<TaskInfo>  MirrorTaskManagerLite::GetTaskInfo(u32 streamId, u32 taskId) const
{
    TaskInfoQueue *queue = nullptr;
    try {
        queue = GetQueue(streamId);
    } catch (HcclException &e) {
        HCCL_ERROR("Hccl exception %s was caught.", e.what());
        return nullptr;
    }

    auto FindTask = [taskId](const std::shared_ptr<TaskInfo> &taskInfo) {
        return taskInfo->taskId_ == taskId;
    };

    auto task = *queue->Find(FindTask);
    if (task == *queue->End()) {
        return nullptr;
    };

    HCCL_INFO("[MirrorTaskManagerLite][GetTaskInfo]find streamdId(sqId)[%u] taskId(sqeId)[%u]", streamId, taskId);

    return *task;
}

TaskInfoQueueMap::iterator MirrorTaskManagerLite::Begin()
{
    return queueMap_.begin();
}

TaskInfoQueueMap::iterator MirrorTaskManagerLite::End()
{
    return queueMap_.end();
}

MirrorTaskManagerLite::~MirrorTaskManagerLite()
{
}

} // namespace Hccl