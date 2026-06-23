/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "roce_transport_lite_impl.h"
#include "binary_stream.h"
#include "log.h"

namespace Hccl {

RoceTransportLiteImpl::RoceTransportLiteImpl(std::vector<char> &uniqueId)
{
    Init(uniqueId);
}

RoceTransportLiteImpl::~RoceTransportLiteImpl()
{
}

void RoceTransportLiteImpl::Init(std::vector<char> &uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    u32 type;
    binaryStream >> type;
    binaryStream >> notifyNum_;
    binaryStream >> bufferNum_;
    binaryStream >> connNum_;

    std::vector<char> locNotifyUniqueIds;
    binaryStream >> locNotifyUniqueIds;
    ParseLocNotifyVec(locNotifyUniqueIds);

    std::vector<char> rmtNotifyUniqueIds;
    binaryStream >> rmtNotifyUniqueIds;
    ParseRmtNotifyVec(rmtNotifyUniqueIds);

    std::vector<char> notifyValueBufferUniqueIds;
    binaryStream >> notifyValueBufferUniqueIds;
    ParseNotifyValueBuffer(notifyValueBufferUniqueIds);

    std::vector<char> locBufferUniqueIds;
    binaryStream >> locBufferUniqueIds;
    ParseLocBufferVec(locBufferUniqueIds);

    std::vector<char> rmtBufferUniqueIds;
    binaryStream >> rmtBufferUniqueIds;
    ParseRmtBufferVec(rmtBufferUniqueIds);

    std::vector<char> connUniqueIds;
    binaryStream >> connUniqueIds;
    ParseConnVec(connUniqueIds);
}

void RoceTransportLiteImpl::ParseLocNotifyVec(std::vector<char> &data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 notifySizePerDto = data.size() / notifyNum_;

    for (u32 idx = 0; idx < notifyNum_; idx++) {
        auto              start = data.begin() + idx * notifySizePerDto;
        auto              end   = start + notifySizePerDto;
        std::vector<char> dto(start, end);
        localNotifies_.push_back(std::make_unique<NotifyLite>(dto));
        HCCL_INFO("locNotify idx=%u, %s", idx, localNotifies_.back()->Describe().c_str());
    }
}

void RoceTransportLiteImpl::ParseRmtNotifyVec(std::vector<char> &data)
{
    if (notifyNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] notifyNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / notifyNum_;
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse remote notify num=%u, sizePerDto=%u",
        __func__, notifyNum_, rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    remoteNotifies_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < notifyNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        remoteNotifies_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseNotifyValueBuffer(std::vector<char> &data)
{
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse notify value buffer", __func__);

    BinaryStream binaryStream(data);
    u64 addr;
    u64 size;
    u32 lkey;
    binaryStream >> addr;
    binaryStream >> size;
    binaryStream >> lkey;
    notifyValueBuffer_ = std::make_unique<RmaBufferLite>(addr, size, lkey);
}

void RoceTransportLiteImpl::ParseLocBufferVec(std::vector<char> &data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 locBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse local buffer num=%u, sizePerDto=%u",
        __func__, bufferNum_, locBufferSizePerDto);

    BinaryStream binaryStream(data);
    locBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 lkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> lkey;
        RmaBufferLite rdmaBufLite(addr, size, lkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        locBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseRmtBufferVec(std::vector<char> &data)
{
    if (bufferNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] bufferNum is 0", __func__);
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / bufferNum_;
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse remote buffer num=%u, sizePerDto=%u",
        __func__, bufferNum_, rmtBufferSizePerDto);

    BinaryStream binaryStream(data);
    rmtBufferVec_.clear();
    u64 addr;
    u64 size;
    u32 rkey;
    for (u32 idx = 0; idx < bufferNum_; idx++) {
        binaryStream >> addr;
        binaryStream >> size;
        binaryStream >> rkey;
        RmtRmaBufferLite rdmaBufLite(addr, size, rkey);
        HCCL_INFO("idx=%u, %s", idx, rdmaBufLite.Describe().c_str());
        rmtBufferVec_.emplace_back(rdmaBufLite);
    }
}

void RoceTransportLiteImpl::ParseConnVec(std::vector<char> &data)
{
    if (connNum_ == 0) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] connNum is 0", __func__);
        return;
    }

    u32 connSizePerDto = data.size() / connNum_;
    HCCL_INFO("[RoceTransportLiteImpl::%s] Parse conn num=%u, sizePerDto=%u",
        __func__, connNum_, connSizePerDto);
    for (u32 idx = 0; idx < connNum_; idx++) {
        auto              start = data.begin() + idx * connSizePerDto;
        auto              end   = start + connSizePerDto;
        std::vector<char> connUniqueId(start, end);
        connUniqueIdVec_.emplace_back(connUniqueId);
        std::unique_ptr<RdmaConnLiteV2> connLite;
        connLite = std::make_unique<RdmaConnLiteV2>(connUniqueId);
        HCCL_INFO("[RoceTransportLiteImpl::%s] idx=%u, %s", __func__, idx, connLite->Describe().c_str());
        connVec_.emplace_back(std::move(connLite));
    }
}

RmaBufSliceLite RoceTransportLiteImpl::GetRmaBufSlicelite(const RmaBufferLite &lite) const
{
    return RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetLkey(), 0);
}

RmaBufSliceLite RoceTransportLiteImpl::GetNotifySlicelite(u32 index) const
{
    (void)index;
    return RmaBufSliceLite(
        notifyValueBuffer_->GetAddr(), 
        notifyValueBuffer_->GetSize(), 
        notifyValueBuffer_->GetLkey(), 0);
}

RmtRmaBufSliceLite RoceTransportLiteImpl::GetRmtRmaBufSliceLite(const Buffer &rmtBuf) const
{
    for (auto &it : rmtBufferVec_) {
        Buffer buf(it.GetAddr(), it.GetSize());
        if (buf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize())) {
            return RmtRmaBufSliceLite(rmtBuf.GetAddr(), rmtBuf.GetSize(), it.GetRkey(), 0, 0, UINT32_MAX);
        }
    }
    MACRO_THROW(InternalException, StringFormat("%s is not in current transport", rmtBuf.Describe().c_str()));
}

RmtRmaBufSliceLite RoceTransportLiteImpl::GetRmtNotifySliceLite(u32 index) const
{
    auto &lite = remoteNotifies_[index];
    return RmtRmaBufSliceLite(lite.GetAddr(), lite.GetSize(), lite.GetRkey(), 0, 0, UINT32_MAX);
}

std::string RoceTransportLiteImpl::Describe() const
{
    std::string desc = "RoceTransportLiteImpl[";

    u32 idx = 0;
    desc += "localNotifies=[";
    for (auto &it : localNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], remoteNotifies=[";
    for (auto &it : remoteNotifies_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], locBufferVec=[";
    for (auto &it : locBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], rmtBufferVec=[";
    for (auto &it : rmtBufferVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], connVec=[";
    for (auto &it : connVec_) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    desc += "]]";
    return desc;
}

HcclResult RoceTransportLiteImpl::BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite &rmaBufferLite)
{
    HCCL_INFO("[RoceTransportLiteImpl::%s] start to find addr[0x%llx], size[0x%llx] in locBufferVec, whose size is %zu. ",
        __func__, addr, size, locBufferVec_.size());
    
    if (locBufferVec_.empty()) {
        HCCL_ERROR("[RoceTransportLiteImpl::%s] locBufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto &it : locBufferVec_) {
        Buffer iterBuf(it.GetAddr(), it.GetSize());
        if (iterBuf.Contains(addr, size)) {
            rmaBufferLite = RmaBufferLite(addr, size, it.GetLkey());
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        HCCL_WARNING("[RoceTransportLiteImpl::%s] addr[0x%llx], size[0x%llx] not in any range of locBufferVec. The token of the first locBuffer is used.",
            __func__, addr, size);
        rmaBufferLite = RmaBufferLite(addr, size, locBufferVec_[0].GetLkey());
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
}

void RoceTransportLiteImpl::Write(const RmaBufferLite &loc, const Buffer &rmt, const StreamLite &stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;

    // Post Wqe && return dbValue
    connVec_[0]->Write(
        GetRmaBufSlicelite(loc), GetRmtRmaBufSliceLite(rmt), dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);
}

void RoceTransportLiteImpl::WriteWithNotify(const RmaBufferLite &loc, const Buffer &rmt, const WithNotifyIn &withNotify, const StreamLite &stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;
    u32 notifyIdx = withNotify.index_;

    // Post Wqe && return dbValue
    connVec_[0]->WriteWithNotify(
        GetRmaBufSlicelite(loc), GetRmtRmaBufSliceLite(rmt),
        GetNotifySlicelite(notifyIdx), GetRmtNotifySliceLite(notifyIdx), dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);
}

void RoceTransportLiteImpl::Post(u32 index, const StreamLite &stream)
{
    u64 dbAddr = 0;
    u64 dbValue = 0;

    // Post Wqe && return dbValue
    connVec_[0]->Write(
        GetNotifySlicelite(index), GetRmtNotifySliceLite(index), dbAddr, dbValue);

    // Ring Doorbell
    BuildRdmaDbSendTask(stream, dbAddr, dbValue);
}

void RoceTransportLiteImpl::WaitWithTimeout(u32 index, const StreamLite &stream, u32 timeout)
{
    auto notifyId = localNotifies_[index]->GetId();
    BuildNotifyWaitTask(notifyId, stream, timeout);
}

// 下发Rtsq sqe, 敲DB
void RoceTransportLiteImpl::BuildRdmaDbSendTask(const StreamLite &stream, u64 remoteAddr, u64 dbValue)
{
    stream.GetRtsq()->RdmaDbSend(remoteAddr, dbValue);
}

// 下发Rtsq sqe, NotifyWait
void RoceTransportLiteImpl::BuildNotifyWaitTask(u32 notifyId, const StreamLite &stream, u32 timeout) const
{
    stream.GetRtsq()->NotifyWait(notifyId, timeout);
}
} // namespace Hccl
