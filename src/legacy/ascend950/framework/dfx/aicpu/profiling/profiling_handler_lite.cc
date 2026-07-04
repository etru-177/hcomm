/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "profiling_handler_lite.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sal.h"
#include "task_info.h"
#include "task_param.h"
#include "communicator_impl_lite.h"
namespace Hccl {

static constexpr u32 aging = 1;
constexpr std::uint32_t HCCLINFO_REPORT_BATCH_NUM = 2;
ProfilingHandlerLite ProfilingHandlerLite::instance_;

ProfilingHandlerLite::ProfilingHandlerLite()
{
}

ProfilingHandlerLite::~ProfilingHandlerLite()
{
}

ProfilingHandlerLite &ProfilingHandlerLite::GetInstance()
{
    return instance_;
}

void ProfilingHandlerLite::Init() const
{
}

void ProfilingHandlerLite::ReportHcclOpInfo(const DfxOpInfo &opInfo) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] l0 is false.");
        return;
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] ReportHcclOpInfo start.");
    MsprofAicpuHCCLOPInfo hcclOpInfo {};
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[ProfilingHandlerLite][ReportHcclOpInfo] GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t taskId   = 0U;
    uint32_t streamId = 0;
    if (aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::status_t::AICPU_ERROR_NONE) {
        THROW<InternalException>("[ProfilingHandler] Failed to get task id and stream id.");
    }
    hcclOpInfo.algType  = GetProfHashId(opInfo.algType_.c_str(), opInfo.algType_.length());
    if (taskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        THROW<InvalidParamsException>("[ProfilingHandler] taskId is larger than u32.");
    }
    hcclOpInfo.taskId   = static_cast<uint32_t>(taskId);
    hcclOpInfo.streamId = streamId;
    hcclOpInfo.count = opInfo.op_.dataCount;
    hcclOpInfo.dataType = opInfo.op_.dataType;
    if (opInfo.isIndop_ == true) {
        hcclOpInfo.groupName = GetProfHashId(opInfo.groupName_.c_str(), opInfo.groupName_.length());
    } else {
        CommunicatorImplLite *commImp = static_cast<CommunicatorImplLite *>(opInfo.comm_);
        if (commImp == nullptr) {
            HCCL_WARNING("[ProfilingHandlerLite][%s] commImp is nullptr, groupName set to 0.", __func__);
            hcclOpInfo.groupName = 0;
        } else {
            hcclOpInfo.groupName = GetProfHashId(commImp->GetId().c_str(), commImp->GetId().length());
        }
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] relay:%u, retry:%u, dataType:%s, algType:%u, count:%llu, "
              "groupName:%lu, ranksize:%u, taskId:%u, streamId:%u",
              hcclOpInfo.relay, hcclOpInfo.retry, DataTypeToSerialString(hcclOpInfo.dataType).c_str(), hcclOpInfo.algType, hcclOpInfo.count,
              hcclOpInfo.groupName, hcclOpInfo.ranksize, hcclOpInfo.taskId, hcclOpInfo.streamId);
    // 信息上报
    ReportAdditionInfo(MSPROF_REPORT_AICPU_HCCL_OP_INFO, ProfGetCurCpuTimestamp(), &hcclOpInfo,
                       sizeof(MsprofAicpuHCCLOPInfo));
    HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] ReportHcclOpInfo end.");
}

void ProfilingHandlerLite::ReportHcclTaskDetails(const std::vector<TaskInfo> &taskInfo) const
{
    if (!GetProfL1State()) {
        HCCL_INFO("[ProfilingHandlerLite][ReportHcclTaskDetails] l1 is false.");
        return;
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] ReporttHcclTaskDetails start.");
    uint32_t                batchId = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos[HCCLINFO_REPORT_BATCH_NUM] {};
    for (std::vector<Hccl::TaskInfo>::size_type i = 0; i < taskInfo.size(); i++) {
        auto &taskDetailInfo = taskDetailsInfos[batchId++];
        GetTaskDetailInfos(taskInfo[i], taskDetailInfo);
        DumpTaskDetails(taskDetailInfo, taskInfo[i]);
        // 信息批量上报
        if (batchId == HCCLINFO_REPORT_BATCH_NUM || i == taskInfo.size() - 1) {
            ReportAdditionInfo(MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO, 0, taskDetailsInfos,
                               sizeof(MsprofAicpuHcclTaskInfo) * batchId);
            batchId = 0;
            memset_s(taskDetailsInfos, sizeof(taskDetailsInfos), 0, sizeof(taskDetailsInfos));
        }
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] ReportHcclTaskDetails end.");
}

void ProfilingHandlerLite::GetTaskDetailInfos(const TaskInfo &it, MsprofAicpuHcclTaskInfo &taskDetailsInfos) const 
{
    HCCL_INFO("ProfilingHandlerLite::GetTaskDetailInfos %s", it.taskParam_.Describe().c_str());
    constexpr u32 INVALID_LOCAL_RANK = UINT32_MAX;
    constexpr u32 INVALID_REMOTE_RANK = UINT32_MAX;
    taskDetailsInfos.localRank = INVALID_LOCAL_RANK;
    taskDetailsInfos.remoteRank = INVALID_REMOTE_RANK;
    if (it.dfxOpInfo_ != nullptr) {
        taskDetailsInfos.localRank = it.dfxOpInfo_->op_.myRank;
        std::string nameInfo = GetProfTaskOpNameV2(it.taskParam_.taskType);
        taskDetailsInfos.itemId = GetProfHashId(nameInfo.c_str(), nameInfo.length());
        taskDetailsInfos.cclTag = GetProfHashId(it.dfxOpInfo_->tag_.c_str(), it.dfxOpInfo_->tag_.length());
        if (it.dfxOpInfo_->isIndop_ == true) {
            taskDetailsInfos.groupName = GetProfHashId(it.dfxOpInfo_->groupName_.c_str(), it.dfxOpInfo_->groupName_.length());
            taskDetailsInfos.rankSize = it.dfxOpInfo_->rankSize_;
            HCCL_INFO("ProfilingHandlerLite::GetTaskDetailInfos groupName[%s], rankSize[%u]",
                it.dfxOpInfo_->groupName_.c_str(), taskDetailsInfos.rankSize);
        } else if (it.dfxOpInfo_->comm_ != nullptr) {
            CommunicatorImplLite *commImp = static_cast<CommunicatorImplLite *>(it.dfxOpInfo_->comm_);
            taskDetailsInfos.groupName = GetProfHashId(commImp->GetId().c_str(), commImp->GetId().length());
            taskDetailsInfos.rankSize = commImp->GetRankSize();
            HCCL_INFO("ProfilingHandlerLite::GetTaskDetailInfos groupName[%s], rankSize[%u]",
                it.dfxOpInfo_->groupName_.c_str(), taskDetailsInfos.rankSize);
        }
    } else {
        HCCL_WARNING("[ProfilingHandlerLite::GetTaskDetailInfos] dfxOpInfo_ is nullptr!");
    }
    taskDetailsInfos.remoteRank = it.remoteRank_;
    taskDetailsInfos.stage = 0;
    if (it.taskParam_.taskType == TaskParamType::TASK_SDMA || it.taskParam_.taskType == TaskParamType::TASK_RDMA
        || it.taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || it.taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
        || it.taskParam_.taskType == TaskParamType::TASK_UB) {
        taskDetailsInfos.srcAddr  = static_cast<u64>(reinterpret_cast<uintptr_t>(it.taskParam_.taskPara.DMA.src));
        taskDetailsInfos.dstAddr  = static_cast<u64>(reinterpret_cast<uintptr_t>(it.taskParam_.taskPara.DMA.dst));
        taskDetailsInfos.dataSize = static_cast<u32>(it.taskParam_.taskPara.DMA.size);
        taskDetailsInfos.notifyID = it.taskParam_.taskPara.DMA.notifyID;
        taskDetailsInfos.linkType = static_cast<uint16_t>(it.taskParam_.taskPara.DMA.linkType);
    } else if (it.taskParam_.taskType == TaskParamType::TASK_REDUCE_INLINE
               || it.taskParam_.taskType == TaskParamType::TASK_REDUCE_TBE
               || it.taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
               || it.taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        taskDetailsInfos.srcAddr  = static_cast<u64>(reinterpret_cast<uintptr_t>(it.taskParam_.taskPara.Reduce.src));
        taskDetailsInfos.dstAddr  = static_cast<u64>(reinterpret_cast<uintptr_t>(it.taskParam_.taskPara.Reduce.dst));
        taskDetailsInfos.dataSize = static_cast<u32>(it.taskParam_.taskPara.Reduce.size);
        taskDetailsInfos.notifyID = it.taskParam_.taskPara.Reduce.notifyID;
        taskDetailsInfos.dataType = static_cast<uint16_t>(it.taskParam_.taskPara.Reduce.dataType);
        taskDetailsInfos.linkType = static_cast<uint16_t>(it.taskParam_.taskPara.Reduce.linkType);
        taskDetailsInfos.opType   = it.taskParam_.taskPara.Reduce.reduceOp;
    } else if (it.taskParam_.taskType == TaskParamType::TASK_NOTIFY_RECORD
               || it.taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        taskDetailsInfos.notifyID = it.taskParam_.taskPara.Notify.notifyID;
    }
    taskDetailsInfos.timeStamp         = ProfGetCurCpuTimestamp();
    taskDetailsInfos.durationEstimated = 0;
    taskDetailsInfos.taskId            = it.taskId_;
    taskDetailsInfos.streamId          = it.streamId_;
    taskDetailsInfos.planeID           = 0;
    taskDetailsInfos.transportType     = static_cast<int32_t>(SimpleTaskType::UB);
    taskDetailsInfos.role              = static_cast<uint32_t>(TaskRole::DST);
    taskDetailsInfos.workFlowMode      = static_cast<uint32_t>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void ProfilingHandlerLite::DumpTaskDetails(const MsprofAicpuHcclTaskInfo &taskDetailsInfos, const TaskInfo &taskInfo) const
{
    HCCL_INFO("ProfilingHandlerLite::DumpTaskDetails %s", taskInfo.taskParam_.Describe().c_str());
    HCCL_INFO("[ProfilingHandlerLite]ReporttHcclTaskDetails data is: itemId[%llu], cclTag[%llu], groupName[%llu], "
              "remoteRank[%u], rankSize[%u], stage[%u], taskType[%s], srcAddr[%llu], dstAddr[%llu], "
              "dataSize[%llu], notifyID[%llu], dataType[%s],linkType[%u], timeStamp[%llu], durationEstimated[%f], "
              "taskId[%u], streamId[%u], planeID[%u], opType[%s], transportType[%d], role[%u], workFlowMode[%u]",
              taskDetailsInfos.itemId, taskDetailsInfos.cclTag, taskDetailsInfos.groupName, taskDetailsInfos.remoteRank,
              taskDetailsInfos.rankSize, taskDetailsInfos.stage, taskInfo.taskParam_.taskType.Describe().c_str(), taskDetailsInfos.srcAddr,
              taskDetailsInfos.dstAddr, taskDetailsInfos.dataSize, taskDetailsInfos.notifyID, DataTypeToSerialString( taskDetailsInfos.dataType).c_str(),
              taskDetailsInfos.linkType, taskDetailsInfos.timeStamp, taskDetailsInfos.durationEstimated,
              taskDetailsInfos.taskId, taskDetailsInfos.streamId, taskDetailsInfos.planeID, OpTypeToSerialString(taskDetailsInfos.opType).c_str(),
              static_cast<int>(taskDetailsInfos.transportType), taskDetailsInfos.role, taskDetailsInfos.workFlowMode);
}

void ProfilingHandlerLite::ReportMainStreamTask(const FlagTaskInfo &flagTaskInfo) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[ProfilingHandlerLite][ReportMainStreamTask] l0 is false.");
        return;
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportMainStreamTask] ReportMainStreamTask start.");
    MsprofAicpuHcclMainStreamTask flagtask {};
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[ProfilingHandlerLite][ReportMainStreamTask] aicpu::GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t aicpuKernelTaskId   = 0U;
    uint32_t aicpuKernelStreamId = 0;
    if (aicpu::GetTaskAndStreamId(aicpuKernelTaskId, aicpuKernelStreamId) != aicpu::status_t::AICPU_ERROR_NONE) {
        THROW<InternalException>("[ProfilingHandler] Failed to get task id and stream id.");
    }
    // flagTaskInfo.taskId的高16位填到flagtask.taskId，低16位填到flagtask.streamId
    flagtask.taskId = static_cast<uint16_t>(flagTaskInfo.taskId >> 16);
    flagtask.streamId = static_cast<uint16_t>(flagTaskInfo.taskId);
    flagtask.type = flagTaskInfo.type;

    if (aicpuKernelTaskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        THROW<InvalidParamsException>("[ProfilingHandler] aicpuKernelTaskId is larger than u32.");
    }
    // aicpuKernelTaskId的高16位填到flagtask.aicpuTaskId，低16位填到flagtask.aicpuStreamId
    HCCL_INFO("[ProfilingHandlerLite][kernelTask] aicpuKernelTaskId %lu. aicpuKernelStreamId %u", aicpuKernelTaskId, aicpuKernelStreamId);
    uint32_t aicpuKernelTaskIdLow32 = static_cast<uint32_t>(aicpuKernelTaskId);
    flagtask.aicpuTaskId = static_cast<uint16_t>(aicpuKernelTaskIdLow32 >> 16);
    flagtask.aicpuStreamId = static_cast<uint16_t>(aicpuKernelTaskIdLow32);
    HCCL_INFO("[ProfilingHandlerLite][ReportMainStreamTask] streamId:%u, taskId:%u, type:%u,"
              "aicpuStreamId:%u, aicpuTaskId:%u",
              flagtask.streamId, flagtask.taskId, flagtask.type, flagtask.aicpuStreamId, flagtask.aicpuTaskId);
    // 信息上报
    ReportAdditionInfo(MSPROF_REPORT_AICPU_HCCL_FLAG_TASK, ProfGetCurCpuTimestamp(), &flagtask,
                               sizeof(MsprofAicpuHcclMainStreamTask));
    HCCL_INFO("[ProfilingHandlerLite][ReportMainStreamTask] ReportMainStreamTask end.");
}

void ProfilingHandlerLite::ReportAdditionInfo(uint32_t type, uint64_t timeStamp, const void *data, int len) const
{
    HCCL_INFO("[ProfilingHandlerLite][ReportAdditionInfo] ReportAdditionInfo start.");
    MsprofAdditionalInfo reporterData{};
    reporterData.level     = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type      = type;
    reporterData.threadId  = SalGetTid();
    reporterData.dataLen   = len;
    reporterData.timeStamp = timeStamp;
    s32 sret               = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    if (sret != EOK) {
        THROW<InternalException>("[ProfilingHandlerLite] memcpy failed, errorno[%d], len[%u], data[%u]", sret, len, sizeof(reporterData.data));
    }
    HCCL_INFO("[ProfilingHandlerLite][ReportAdditionInfo] level :%u, type:%u, threadId:%u, dataLen:%u, timeStamp:%u",
              reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen,
              reporterData.timeStamp);
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofReportAdditionalInfo == nullptr) {
            HCCL_WARNING("[ProfilingHandlerLite][ReportAdditionInfo] AdprofReportAdditionalInfo is nullptr.");
            return;
        }
        if (AdprofReportAdditionalInfo(aging, &reporterData, sizeof(MsprofAdditionalInfo)) != 0) {
            THROW<InternalException>("[ProfilingHandler] AdprofReportAdditionalInfo failed.");
        }
    } else {
        if (MsprofReportAdditionalInfo == nullptr) {
            HCCL_WARNING("[ProfilingHandlerLite][ReportAdditionInfo] MsprofReportAdditionalInfo is nullptr.");
            return;
        }
        if (MsprofReportAdditionalInfo(aging, &reporterData, sizeof(MsprofAdditionalInfo)) != 0) {
            THROW<InternalException>("[ProfilingHandler] MsprofReportAdditionalInfo failed.");
        }
    }
    HCCL_INFO("[ProfilingHandlerLite] ReportAdditionInfo with additionInfoType[%u] successfully", type);
}

void ProfilingHandlerLite::UpdateProfSwitch()
{
    HCCL_INFO("[ProfilingHandlerLite][UpdateProfSwitch] UpdateProfSwitch start.");
    IsL1fromOffToOn();
    IsProfSwitchOn(ProfilingLevel::L0);
    IsProfSwitchOn(ProfilingLevel::L1);
    HCCL_INFO("[ProfilingHandlerLite][UpdateProfSwitch] UpdateProfSwitch end.");
}

bool ProfilingHandlerLite::IsProfOn(uint64_t feature) const
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofCheckFeatureIsOn == nullptr) {
            return false;
        }
        return AdprofCheckFeatureIsOn(feature) > 0;
    } else {
        if (feature == ADPROF_TASK_TIME_L1) {
            return enableHcclL1_;
        } else if (feature == ADPROF_TASK_TIME_L0) {
            return enableHcclL0_;
        }
    }
 
    return false;
}

bool ProfilingHandlerLite::IsProfSwitchOn(ProfilingLevel level)
{
    bool res = false;
    if (level == ProfilingLevel::L0) {
        res           = IsProfOn(ADPROF_TASK_TIME_L0);
        enableHcclL0_ = res;
    } else if (level == ProfilingLevel::L1) {
        res           = IsProfOn(ADPROF_TASK_TIME_L1);
        enableHcclL1_ = res;
    }
    return res;
}

bool ProfilingHandlerLite::IsL1fromOffToOn()
{
    if (((!GetProfL1State()) && IsProfSwitchOn(ProfilingLevel::L1))) {
        HCCL_INFO("Profiling L1 switch form off to on.");
        return true;
    }
    return false;
}

void ProfilingHandlerLite::SetProL1On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL1_ = val;
}
 
void ProfilingHandlerLite::SetProL0On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL0_ = val;
}

uint64_t ProfilingHandlerLite::GetProfHashId(const char *name, uint32_t len) const
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty.  name:%s, len:%u", name, len);
        return INVALID_U64;
    }
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofGetHashId == nullptr) {
            return INVALID_U64;
        }
        return AdprofGetHashId(name, len);
    } else {
        if (MsprofStr2Id == nullptr) {
            return INVALID_U64;
        }
        return MsprofStr2Id(name, len);
    }
}

} // namespace Hccl