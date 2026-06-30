/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_primitives_impl.h"

#include "ccu_log.h"
#include "hcom_common.h"

#include "ccu_kernel_mgr.h"

//Alloc 相关接口
CcuResult CcuVariableAlloc(CcuVariableHandle *varHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->VariableAlloc(varHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAlloc(CcuAddressHandle *addrHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAlloc(addrHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuEventAlloc(CcuEventHandle *eventHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->EventAlloc(eventHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuBufferAlloc(CcuBufferHandle *bufHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->BufferAlloc(bufHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalAddrAlloc(CcuLocalAddrHandle *localAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalAddrAlloc(localAddrHandle, addrHandle, tokenHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuRemoteAddrAlloc(CcuRemoteAddrHandle *remoteAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->RemoteAddrAlloc(remoteAddrHandle, addrHandle, tokenHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuBlockVariableAlloc(CcuVariableHandle *varHandles, uint32_t count)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->BlockVariableAlloc(varHandles, count));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuBlockEventAlloc(CcuEventHandle *eventHandles, uint32_t count)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->BlockEventAlloc(eventHandles, count));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuBlockBufferAlloc(CcuBufferHandle *bufHandles, uint32_t count)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->BlockBufferAlloc(bufHandles, count));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuVariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle *varHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->VariableCreateByChannel(channel, varIndex, varHandle));
    return CcuResult::CCU_SUCCESS;
}

//Variable操作类 相关接口
CcuResult CcuVariableAssignImm(CcuVariableHandle resVar, uint64_t immediate)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->VariableAssignImm(resVar, immediate));
    
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuVariableAssignVar(CcuVariableHandle dstVarHandle, CcuVariableHandle srcVarHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->VariableAssignVar(dstVarHandle, srcVarHandle));
    
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuVariableAddVarToVar(CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->VariableAddVarToVar(resVar, varA, varB));
    
    return CcuResult::CCU_SUCCESS;
}

/*
Address 相关接口
*/
CcuResult CcuAddressAssignImm(CcuAddressHandle addr, uint64_t immediate)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAssignImm(addr, immediate));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAssignAddr(CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAssignAddr(dstAddrHandle, srcAddrHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAssignVar(CcuAddressHandle addr, CcuVariableHandle var)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAssignVar(addr, var));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAddVarToAddr(CcuAddressHandle resAddr, CcuAddressHandle lhsAddr, CcuVariableHandle rhsVar)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAddVarToAddr(resAddr, lhsAddr, rhsVar));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAddAddrToAddr(CcuAddressHandle resAddr, CcuAddressHandle addrA, CcuAddressHandle addrB)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAddAddrToAddr(resAddr, addrA, addrB));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAddressAddAssignVar(CcuAddressHandle addr, CcuVariableHandle var)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->AddressAddAssignVar(addr, var));
    return CcuResult::CCU_SUCCESS;
}

//参数加载类 相关接口
CcuResult CcuLoadArg(CcuVariableHandle varHandle, uint32_t argId)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoadArg(varHandle, argId));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoadVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    if (num == 0) {
        HCCL_ERROR("[CcuLoadVar] invalid args, num[%u]", num);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoadVar(addr, varHandle, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoadVarFromVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    if (num == 0) {
        HCCL_ERROR("[CcuLoadVarFromVarAddr] invalid args, num[%u]", num);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->CcuLoadVarFromVarAddr(addrHandle, varHandle, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuStoreVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    if (num == 0) {
        HCCL_ERROR("[CcuStoreVar] invalid args, num[%u]", num);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->StoreVar(addr, varHandle, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuStoreVarToVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    if (num == 0) {
        HCCL_ERROR("[CcuStoreVarToVarAddr] invalid args, num[%u]", num);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->CcuStoreVarToVarAddr(addrHandle, varHandle, num));
    return CcuResult::CCU_SUCCESS;
}

//Event信号同步类 相关接口
CcuResult CcuEventRecord(CcuEventHandle eventHandle, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->EventRecord(eventHandle, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuEventWait(CcuEventHandle eventHandle, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->EventWait(eventHandle, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->NotifyRecord(channel, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask)  
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->NotifyWait(channel, localNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuWriteVariableWithNotify(ChannelHandle channel, CcuVariableHandle varHandle,uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WriteVariableWithNotify(channel, varHandle, remoteVarIdx, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLocalNotifyRecord(const char *notifyTag, uint16_t mask)
{
    CCU_CHK_PTR_NULL(notifyTag);
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalNotifyRecord(notifyTag, mask));
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLocalNotifyWait(const char *notifyTag, uint16_t mask)
{
    CCU_CHK_PTR_NULL(notifyTag);
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalNotifyWait(notifyTag, mask));
    return CcuResult::CCU_SUCCESS;
}

//本地数据拷贝 相关接口
CcuResult CcuLocalCopyMemToMem(
    CcuLocalAddrHandle dst, CcuLocalAddrHandle src,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalCopyMemToMem(dst, src, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalCopyMemToBuffer(
    CcuBufferHandle dst, CcuLocalAddrHandle src,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalCopyMemToBuffer(dst, src, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalCopyBufferToMem(
    CcuLocalAddrHandle dst, CcuBufferHandle src,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalCopyBufferToMem(dst, src, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}
//本地reduce 相关接口
CcuResult CcuLocalMemReduce(CcuLocalAddrHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalMemReduce(dst, src, len, dataType, opType, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalBufferReduce(CcuBufferHandle* buffers, uint32_t count, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType, CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    if (buffers == nullptr || count == 0) {
        HCCL_ERROR("[CcuLocalBufferReduce] invalid args, buffers[%p] count[%u]", buffers, count);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LocalBufferReduce(buffers, count, dataType, outputDataType, opType, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

/*========== 远端数据传输操作 ==========*/
CcuResult CcuReadMemToMem(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->ReadMemToMem(channel, localHandle, remoteHandle, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuReadMemToBuffer(
    ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->ReadMemToBuffer(channel, localHandle, remoteHandle, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuReadMemToMemReduce(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle len, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->ReadMemToMemReduce(channel, localHandle, remoteHandle, len, dataType, opType, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuWriteMemToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remote, CcuLocalAddrHandle local, 
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WriteMemToMem(channel, remote, local, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuWriteBufferToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle,
    CcuVariableHandle len, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WriteBufferToMem(channel, remoteHandle, localHandle, len, event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuWriteMemToMemReduce(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle len, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle event, uint16_t mask)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WriteMemToMemReduce(channel, remoteHandle, localHandle, len, dataType, opType, event, mask));
    return CcuResult::CCU_SUCCESS;
}

/*========== 控制流操作 ==========*/
CcuResult CcuIfBegin(CcuVariableHandle var, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->IfBegin(var, immediate, condType, label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuIfElse(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->IfElse(label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuIfEnd(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->IfEnd(label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuFlushPendingIfs()
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    kernel->FlushClosablePendingIfs();
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuWhileBegin(CcuVariableHandle var, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WhileBegin(var, immediate, condType, label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuWhileEnd(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->WhileEnd(label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuDoWhileBegin(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->DoWhileBegin(label));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuDoWhileEnd(CcuVariableHandle var, uint64_t immediate,
    CcuConditionType condType, const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->DoWhileEnd(var, immediate, condType, label));

    return CcuResult::CCU_SUCCESS;
}

/*========== 函数调用操作 ==========*/
CcuResult CcuFuncBlockLookup(const void *funcPtr, uint64_t *outHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->FuncBlockLookup(funcPtr, outHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuFuncBlockBegin(const void *funcPtr, uint64_t *outHandle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->FuncBlockBegin(funcPtr, outHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuFuncBlockEnd(uint64_t handle)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->FuncBlockEnd(handle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuFuncDefineInArg(uint64_t handle, CcuVariableHandle formal)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->FuncDefineInArg(handle, formal));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuFuncCall(uint64_t handle, const CcuVariableHandle *inArgs, uint32_t numIn)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->FuncCall(handle, inArgs, numIn));
    return CcuResult::CCU_SUCCESS;
}

/*========== 循环操作 ==========*/
CcuResult CcuLoopCreate(CcuLoop *loop)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopCreate(loop));
    return CcuResult::CCU_SUCCESS;
}

CcuResult _CcuLoopBodyEnter(CcuLoop loop)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopBodyEnter(loop));
    return CcuResult::CCU_SUCCESS;
}

CcuResult _CcuLoopBodyExit(CcuLoop loop)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopBodyExit(loop));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoopGroupCreate(CcuLoopGroup *group, uint32_t maxLoopNum,
    const CcuLoopGroupConfig *config)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopGroupCreate(group, maxLoopNum, config));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoopGroupCreateFromVar(CcuLoopGroup *group, uint32_t maxLoopNum,
    CcuVariableHandle parallelVar, CcuVariableHandle offsetVar)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopGroupCreateFromVar(group, maxLoopNum, parallelVar, offsetVar));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoopGroupAddLoop(CcuLoopGroup group,
    CcuLoop loop, const CcuLoopConfig *config)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopGroupAddLoop(group, loop, config));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLoopGroupAddLoopFromVar(CcuLoopGroup group,
    CcuLoop loop, CcuVariableHandle loopParamVar)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    CCU_CHK_PTR_NULL(kernel);
    CCU_CHK_RET(kernel->LoopGroupAddLoopFromVar(group, loop, loopParamVar));
    return CcuResult::CCU_SUCCESS;
}

//控制流标签栈 C 接口（_CcuIfStack* / _CcuDoWhileStack*）
 
void _CcuIfStackPush(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    if (kernel == nullptr) {
        HCCL_ERROR("[_CcuIfStackPush] no current kernel, label=%s",
            label != nullptr ? label : "(null)");
        return;
    }
    kernel->IfLabelStackPush(label);
}

void _CcuIfStackMarkBodyDone()
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    if (kernel == nullptr) {
        HCCL_ERROR("[_CcuIfStackMarkBodyDone] no current kernel");
        return;
    }
    kernel->IfLabelStackMarkBodyDone();
}

const char *_CcuIfStackPopForElse()
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    if (kernel == nullptr) {
        HCCL_ERROR("[_CcuIfStackPopForElse] no current kernel");
        return nullptr;
    }
    return kernel->IfLabelStackPopForElse();
}

void _CcuDoWhileStackPush(const char *label)
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    if (kernel == nullptr) {
        HCCL_ERROR("[_CcuDoWhileStackPush] no current kernel, label=%s",
            label != nullptr ? label : "(null)");
        return;
    }
    kernel->DoWhileLabelStackPush(label);
}

const char *_CcuDoWhileStackPopForWhile()
{
    const uint32_t devLogicId = HcclGetThreadDeviceId();
    auto kernel = hcomm::CcuKernelMgr::GetInstance(devLogicId).GetCurrentKernel();
    if (kernel == nullptr) {
        // 见上方注释：CCU_WHILE 每次都会调本函数做模式判别，保持沉默。
        return nullptr;
    }
    return kernel->DoWhileLabelStackPopForWhile();
}
