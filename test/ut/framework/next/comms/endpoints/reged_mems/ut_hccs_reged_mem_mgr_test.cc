/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <string>

#include "hccl_network.h"
#include "local_ipc_rma_buffer.h"

#define private public
#include "hccs_reged_mem_mgr.h"
#undef private

using namespace hcomm;

class HccsRegedMemMgrTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

template <typename Mgr, typename Key>
static uint64_t GetRef(Mgr& mgr, const Key& key)
{
    for (auto it = mgr.Begin(); it != mgr.End(); ++it) {
        if (it->first == key) {
            return it->second.ref;
        }
    }
    return 0;
}


TEST_F(HccsRegedMemMgrTest, ut_HccsRegedMemMgr_When_LocalIpcInitFails_Expect_ReturnError)
{
    MOCKER_CPP(&hccl::LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_E_INTERNAL));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    HccsRegedMemMgr mgr(reinterpret_cast<HcclNetDevCtx>(&netCtx));
    HcommMem mem{};
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void *>(0xA0000ULL);
    mem.size = 4096U;
    void *handle = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(mem, "init_fail", &handle), HCCL_E_INTERNAL);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HccsRegedMemMgrTest, ut_HccsRegedMemMgr_When_AliasParentMissing_Expect_UnregisterChildNotFound)
{
    MOCKER_CPP(&hccl::LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    HccsRegedMemMgr mgr(reinterpret_cast<HcclNetDevCtx>(&netCtx));
    auto localIpcRmaBufferMgr = netCtx.GetlocalIpcRmaBufferMgr();

    HcommMem mem{};
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void *>(0xB0000ULL);
    mem.size = 4096U;
    void *parentHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(mem, "parent", &parentHandle), HCCL_SUCCESS);
    ASSERT_NE(parentHandle, nullptr);

    void *childHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(mem, "child", &childHandle), HCCL_SUCCESS);
    ASSERT_NE(childHandle, nullptr);
    EXPECT_NE(childHandle, parentHandle);
    auto *childBuffer = static_cast<hccl::LocalIpcRmaBuffer *>(childHandle);
    EXPECT_TRUE(childBuffer->IsAlias());

    auto *parentBuffer = static_cast<hccl::LocalIpcRmaBuffer *>(parentHandle);
    hccl::BufferKey<uintptr_t, u64> parentKey(
        reinterpret_cast<uintptr_t>(parentBuffer->GetAddr()), static_cast<u64>(parentBuffer->GetSize()));
    ASSERT_EQ(mgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
    (void)localIpcRmaBufferMgr->Del(parentKey);

    EXPECT_EQ(mgr.UnregisterMemory(childHandle), HCCL_E_NOT_FOUND);
}

// 父+子集注册 → 先解注册子(alias) → 再解注册父
TEST_F(HccsRegedMemMgrTest, ut_HccsRegedMemMgr_When_ParentChild_Expect_UnregisterChildFirst)
{
    MOCKER_CPP(&hccl::LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    HccsRegedMemMgr mgr(reinterpret_cast<HcclNetDevCtx>(&netCtx));
    auto localIpcRmaBufferMgr = netCtx.GetlocalIpcRmaBufferMgr();

    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x3000;
    mem0.size = 8192;
    void *h0 = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(mem0, "parent", &h0), HCCL_SUCCESS);
    ASSERT_NE(h0, nullptr);

    auto* parentBuf = static_cast<hccl::LocalIpcRmaBuffer*>(h0);
    hccl::BufferKey<uintptr_t, u64> parentKey(
        reinterpret_cast<uintptr_t>(parentBuf->GetAddr()),
        static_cast<u64>(parentBuf->GetSize()));
    EXPECT_TRUE(localIpcRmaBufferMgr->Find(parentKey).first);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 1u);

    // 子集注册（应创建alias buffer）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x3000;
    mem1.size = 1024;
    void *h1 = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(mem1, "child", &h1), HCCL_SUCCESS);
    EXPECT_NE(h1, h0);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 2u);

    // 解注册子（ref 2→1）
    EXPECT_EQ(mgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_TRUE(localIpcRmaBufferMgr->Find(parentKey).first);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 1u);

    // 解注册父（ref 1→0）
    EXPECT_EQ(mgr.UnregisterMemory(h0), HCCL_SUCCESS);
    EXPECT_FALSE(localIpcRmaBufferMgr->Find(parentKey).first);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 0u);
}

// 多次注册同一范围，各自解注册
TEST_F(HccsRegedMemMgrTest, ut_HccsRegedMemMgr_When_MultipleSameRange_Expect_AllSuccess)
{
    MOCKER_CPP(&hccl::LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    HccsRegedMemMgr mgr(reinterpret_cast<HcclNetDevCtx>(&netCtx));
    auto localIpcRmaBufferMgr = netCtx.GetlocalIpcRmaBufferMgr();

    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x8000;
    mem.size = 4096;

    void *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(mem, "t1", &h1), HCCL_SUCCESS);
    auto* parentBuf = static_cast<hccl::LocalIpcRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> parentKey(
        reinterpret_cast<uintptr_t>(parentBuf->GetAddr()),
        static_cast<u64>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 1u);

    EXPECT_EQ(mgr.RegisterMemory(mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 2u);

    EXPECT_EQ(mgr.RegisterMemory(mem, "t3", &h3), HCCL_SUCCESS);
    EXPECT_NE(h3, h1);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 3u);

    EXPECT_EQ(mgr.UnregisterMemory(h3), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 2u);
    EXPECT_EQ(mgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 1u);
    EXPECT_EQ(mgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*localIpcRmaBufferMgr, parentKey), 0u);
}

TEST_F(HccsRegedMemMgrTest, Ut_MemoryExport_When_MemHandleUnregistered_Expect_NotFound)
{
    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    HccsRegedMemMgr mgr(reinterpret_cast<HcclNetDevCtx>(&netCtx));
    EndpointDesc endpointDesc{};
    auto localIpcRmaBuffer = std::make_shared<hccl::LocalIpcRmaBuffer>(
        reinterpret_cast<HcclNetDevCtx>(&netCtx), reinterpret_cast<void *>(0xD1000ULL), 4096U,
        hccl::RmaMemType::DEVICE);
    void *memHandle = localIpcRmaBuffer.get();

    mgr.allRegisteredBuffers_.emplace_back(localIpcRmaBuffer, false);
    mgr.allRegisteredBuffers_.clear();

    void *memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(mgr.MemoryExport(endpointDesc, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_FOUND);
    EXPECT_EQ(memDesc, nullptr);
    EXPECT_EQ(memDescLen, 0U);
}
