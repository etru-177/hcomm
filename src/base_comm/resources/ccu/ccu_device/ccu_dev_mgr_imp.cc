/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dev_mgr_imp.h"

#include "hccl_res.h"
#include "hccl_common.h"
#include "eid_info_mgr.h"

// 支持ccu新老通信域混跑临时添加
#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"
#include "unified_platform/ccu/ccu_device/ccu_res_specs.h"
#include "unified_platform/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "orion_adpt_utils.h"
#include "exception_handler.h"

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * #include "ccu_comp.h"
 * #include "ccu_res_specs.h"
 * #include "ccu_res_batch_allocator.h"
 */

// 引入主板类型查询接口，后续应根据ccu驱动提供的信息用于判断
// 当前先简化修改
#include "./ccu_res_specs.h"

#include "adapter_rts.h"

#include "ccu_types.h"
#include "ccu_log.h"

namespace hcomm {

static std::unordered_map<int32_t, std::shared_ptr<CcuDrvHandle>> ccuDrvHandleMap;
static std::mutex ccuDrvHandleMutex;
static bool ccuDriverInitAgainFlag = false; // 记录每个进程CCU驱动是否重复拉起
static thread_local HcclMainboardId mainBoardType = HcclMainboardId::MAINBOARD_OTHERS; // 记录本卡的主板类型

CcuResult CcuInitFeature(const int32_t devLogicId, std::shared_ptr<CcuDrvHandle> &ccuDrvHandle)
{
    if (devLogicId >= static_cast<int32_t>(MAX_MODULE_DEVICE_NUM)) {
        HCCL_ERROR("[%s] failed, devLogicId[%d] is too large, should be less than %u.",
            __func__, devLogicId, MAX_MODULE_DEVICE_NUM);
        return CcuResult::CCU_E_PARA;
    }

    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    // ccu驱动已重复拉起失败时，直接返回，在锁保护内返回
    if (ccuDriverInitAgainFlag) {
        return CcuResult::CCU_E_DRV_BUSY;
    }

    auto iter = ccuDrvHandleMap.find(devLogicId);
    if (iter != ccuDrvHandleMap.end()) {
        ccuDrvHandle = iter->second;
        HCCL_RUN_INFO("[%s] devLogicId[%d] init ccu feature, handle[0x%llx].",
            __func__, devLogicId, ccuDrvHandle.get());
        return CcuResult::CCU_SUCCESS;
    }

    std::shared_ptr<CcuDrvHandle> drvHandle = nullptr;
    drvHandle.reset(new (std::nothrow) CcuDrvHandle(devLogicId));
    CCU_CHK_PTR_NULL(drvHandle);

    auto ret = drvHandle->Init();
    if (ret == CcuResult::CCU_E_DRV_BUSY) {
        HCCL_RUN_WARNING("[%s] failed but passed, devLogicId[%d] ccu driver has been "
            "inited by another process, this process will not try to init anymore.",
            __func__, devLogicId);
        ccuDriverInitAgainFlag = true; // 记录该进程ccu驱动已拉起失败
        drvHandle = nullptr; // 主动置空触发资源销毁，控制释放时序
        return ret;
    }
    CCU_CHK_RET(ret);

    ccuDrvHandleMap[devLogicId] = drvHandle;
    ccuDrvHandle = ccuDrvHandleMap[devLogicId];
    HCCL_RUN_INFO("[%s] devLogicId[%d] init ccu feature, handle[0x%llx].",
        __func__, devLogicId, ccuDrvHandle.get());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuDeinitFeature(const int32_t devLogicId)
{
    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    auto iter = ccuDrvHandleMap.find(devLogicId);
    if (iter == ccuDrvHandleMap.end()) {
        HCCL_INFO("[%s] passed, ccu feature was not inited, devLogicId[%d].",
            __func__, devLogicId);
        return CcuResult::CCU_SUCCESS;
    }

    auto &ccuDrvHandle = ccuDrvHandleMap[devLogicId];
    if (ccuDrvHandle.use_count() == 1) {
        HCCL_RUN_INFO("[%s] entry, start to deinit ccu feature, "
            "handle[0x%llx] devLogicId[%d].",
            __func__, ccuDrvHandle.get(), devLogicId);
        ccuDrvHandle = nullptr;
        ccuDrvHandleMap.erase(devLogicId);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetDieEnableInfo(int32_t deviceLogicId, uint8_t dieId, bool &enableFlag)
{
    CHK_PRT_RET(dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            __func__, dieId, CCU_MAX_IODIE_NUM, deviceLogicId),
        CcuResult::CCU_E_PARA);
    const auto &dieEnableFlags = Hccl::CcuComponent::GetInstance(deviceLogicId).GetDieEnableFlags();

    enableFlag = dieEnableFlags[dieId];
    return CcuResult::CCU_SUCCESS;
}

inline void ConfigCcuResReqCcuMs(CcuResReq &resReq, uint8_t dieId)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = 8 * 8 * 2;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = 64 * 8 * 2;
    resReq.ckeReq[dieId] = 32;
    resReq.blockCkeReq[dieId] = 8 * 8 * 2;
    resReq.continuousXnReq[dieId] = 400;
    resReq.xnReq[dieId] = 0;
    resReq.gsaReq[dieId] = 400;
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = 2;
}

inline void ConfigCcuResReqCcuSched(CcuResReq &resReq, uint8_t dieId)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = 16;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = 128;
    resReq.ckeReq[dieId] = 32;
    resReq.blockCkeReq[dieId] = 16;
    resReq.continuousXnReq[dieId] = 400;
    resReq.xnReq[dieId] = 0;
    resReq.gsaReq[dieId] = 400;
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = 2;
}

// CCU设备管理对集合通信提供的接口
CcuResult CcuAllocResHandleByInsType(int32_t deviceLogicId,
    CcuInstanceType ccuInsType, CcuResHandle &resHandle)
{
    if (ccuInsType >= CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[%s] failed, error ccu instance type[%d], devLogicId[%d].",
            __func__, ccuInsType, deviceLogicId);
        return CcuResult::CCU_E_PARA;
    }

    std::array<bool, CCU_MAX_IODIE_NUM> dieEnableFlags = {false, false};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        CCU_CHK_RET(CcuGetDieEnableInfo(deviceLogicId, dieId, dieEnableFlags[dieId]));
    }

    if (!dieEnableFlags[0] && !dieEnableFlags[1]) {
        HCCL_ERROR("[%s] failed, all ccu dies are disable, devLogicId[%d].",
            __func__, deviceLogicId);
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuResReq resReq{};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            continue;
        }

        if (ccuInsType == CcuInstanceType::CCU_MS) {
            ConfigCcuResReqCcuMs(resReq, dieId);
        } else {
            ConfigCcuResReqCcuSched(resReq, dieId);
        }
    }

    if (mainBoardType == HcclMainboardId::MAINBOARD_OTHERS) {
        CCU_CHK_RET(CcuGetMainboardId(deviceLogicId, mainBoardType));
    }

    if (mainBoardType == HcclMainboardId::MAINBOARD_PCIE_STD &&
        ccuInsType == CcuInstanceType::CCU_MS) { // 标卡环境下配置CCU_MS拦截报错
        HCCL_ERROR("[%s] ccuInstanceType[%d] not support in %s", __func__,
            ccuInsType, mainBoardType.Describe().c_str());
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    CCU_CHK_RET(CcuDevMgrImp::AllocResHandle(deviceLogicId, resReq, resHandle));

    HCCL_INFO("[%s] succeed, get res handle[%llx], devLogicId[%d]",
        __func__, resHandle, deviceLogicId);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuCheckResource(const int32_t deviceLogicId, const CcuResHandle resHandle, CcuResRepository &resRepo)
{
    CCU_CHK_RET(CcuDevMgrImp::GetResource(deviceLogicId, resHandle, resRepo));
    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle resHandle)
{
    CHK_RET(CcuDevMgrImp::ReleaseResHandle(deviceLogicId, resHandle));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuAllocChannels(const int32_t deviceLogicId, const CcuChannelPara &ccuChannelPara,
    std::vector<CcuChannelInfo> &ccuChannelInfos)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(ccuChannelPara.commAddr, ipAddr)); // 为了打印信息暂时添加
    HCCL_INFO("[%s] new allocation request: deviceLogicId[%d], ipAddr[%s], "
        "channelnum[%u], jettyNum[%u], sqSize[%u].", __func__, deviceLogicId,
        ipAddr.Describe().c_str(), ccuChannelPara.channelNum,
        ccuChannelPara.jettyNum, ccuChannelPara.sqSize);

    uint32_t devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId));

    DevEidInfo eidInfo{};
    CHK_RET(EidInfoMgr::GetInstance(devPhyId).GetEidInfoByAddr(ccuChannelPara.commAddr, eidInfo));
    const uint8_t dieId = static_cast<uint8_t>(eidInfo.dieId);
    const uint32_t feId = eidInfo.funcId;
    ChannelPara para{};
    para.feId = feId;
    para.jettyNum = ccuChannelPara.jettyNum;
    para.sqSize = ccuChannelPara.sqSize;

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocChannels(dieId, para, ccuChannelInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t ccuChannelId)
{
    HCCL_INFO("[%s] new release request: deviceLogicId[%d], dieId[%u], "
        "ccuChannelId[%u].", __func__, deviceLogicId, dieId, ccuChannelId);

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseChannel(dieId, ccuChannelId);
    EXCEPTION_HANDLE_END
    return ret;
}

// 以下为hcomm基础通信内部CCU流程使用的接口
HcclResult CcuDevMgrImp::GetCcuVersion(const int32_t deviceLogicId, CcuVersion &ccuVersion)
{
    EXCEPTION_HANDLE_BEGIN
    ccuVersion = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetCcuVersion();
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDevMgrImp::GetCcuResourceSpaceBufInfo(const int32_t deviceLogicId, const uint8_t dieId,
    uint64_t &addr, uint64_t &size)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceBufInfo(dieId, addr, size);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(const int32_t deviceLogicId, const uint8_t dieId,
    uint64_t &tokenId, uint64_t &tokenValue)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ConfigChannel(const int32_t deviceLogicId, const uint8_t dieId,
    ChannelCfg &cfg)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ConfigChannel(dieId, cfg);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetLoopChannelId(const int32_t deviceLogicId, const uint8_t srcDieId,
    const uint8_t dstDieId, uint32_t &channIdx)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).GetLoopChannelId(srcDieId, dstDieId, channIdx);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetResource(const int32_t deviceLogicId,
    const CcuResHandle handle, CcuResRepository &ccuResRepo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).GetResource(handle, ccuResRepo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq,
    CcuResHandle &handle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).AllocResHandle(resReq, handle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResBatchAllocator::GetInstance(deviceLogicId).ReleaseResHandle(handle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocIns(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, ResInfo &insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocIns(dieId, num, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseIns(const int32_t deviceLogicId, const uint8_t dieId,
    const ResInfo &insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseIns(dieId, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocCke(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, std::vector<ResInfo> &ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocCke(dieId, num, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseCke(const int32_t deviceLogicId, const uint8_t dieId,
    const std::vector<ResInfo> &ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseCke(dieId, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocXn(const int32_t deviceLogicId, const uint8_t dieId,
    const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).AllocXn(dieId, num, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseXn(const int32_t deviceLogicId, const uint8_t dieId,
    const std::vector<ResInfo> &xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).ReleaseXn(dieId, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetMissionKey(const int32_t deviceLogicId, const uint8_t dieId,
    uint32_t &missionKey)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetMissionKey(dieId, missionKey);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId,
    uint32_t &instrNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(deviceLogicId).GetInstructionNum(dieId, instrNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetXnBaseAddr(const uint32_t devLogicId, const uint8_t dieId,
    uint64_t& xnBaseAddr)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuResSpecifications::GetInstance(devLogicId).GetXnBaseAddr(dieId, xnBaseAddr);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CheckDieValid(const char *funcName, const int32_t devLogicId, const uint8_t dieId,
    const std::array<bool, CCU_MAX_IODIE_NUM> &dieEnableFlags)
{
    CHK_PRT_RET(dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].",
            funcName, dieId, CCU_MAX_IODIE_NUM, devLogicId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(!dieEnableFlags[dieId],
        HCCL_ERROR("[%s] failed, dieId[%u] is disable, devLogicId[%d].",
            funcName, dieId, devLogicId),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

bool CcuIsInited(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuIsInited] Input params: deviceLogicId[%d]", deviceLogicId);
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuIsInited]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        false);
    return Hccl::CcuComponent::GetInstance(deviceLogicId).IsInited();
}

HcclResult CcuSetTaskKill(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKill] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKill]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).SetTaskKill();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuSetTaskKillDone(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKillDone] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKillDone]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).SetTaskKillDone();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuCleanTaskKillState] Input params: deviceLogicId[%u]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuCleanTaskKillState]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).CleanTaskKillState();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanDieCkes(const int32_t deviceLogicId, const uint8_t dieId)
{
    HCCL_INFO("[CcuCleanDieCkes] Input params: deviceLogicId[%u], dieId[%u]", deviceLogicId, dieId);
    // 入参校验拦截
    CHK_PRT_RET((deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuCleanDieCkes]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = Hccl::CcuComponent::GetInstance(deviceLogicId).CleanDieCkes(dieId);
    EXCEPTION_HANDLE_END
    return ret;
}

}; // namespace hcomm