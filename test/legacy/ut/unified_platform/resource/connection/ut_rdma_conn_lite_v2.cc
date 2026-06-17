/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "rdma_conn_lite_v2.h"
#include "binary_stream.h"

#define private public

using namespace Hccl;

static std::vector<char> BuildSqUniqueId(const RdmaSqContextLite &sqCtx)
{
    BinaryStream bs;
    bs << sqCtx.qpn;
    bs << sqCtx.sqVa;
    bs << sqCtx.wqeSize;
    bs << sqCtx.depth;
    bs << sqCtx.headAddr;
    bs << sqCtx.tailAddr;
    bs << sqCtx.sl;
    bs << sqCtx.dbVa;
    bs << sqCtx.dbMode;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildCqUniqueId(const RdmaCqContextLite &cqCtx)
{
    BinaryStream bs;
    bs << cqCtx.cqn;
    bs << cqCtx.cqVa;
    bs << cqCtx.cqeSize;
    bs << cqCtx.cqDepth;
    bs << cqCtx.headAddr;
    bs << cqCtx.tailAddr;
    bs << cqCtx.dbVa;
    bs << cqCtx.dbMode;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildRdmaConnLiteV2UniqueId(u32 dmaMode, const RdmaSqContextLite &sqCtx, const RdmaCqContextLite &cqCtx)
{
    BinaryStream binaryStream;
    binaryStream << dmaMode;
    
    std::vector<char> sqUniqueId = BuildSqUniqueId(sqCtx);
    binaryStream << sqUniqueId;
    
    std::vector<char> cqUniqueId = BuildCqUniqueId(cqCtx);
    binaryStream << cqUniqueId;
    
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

static RdmaSqContextLite MakeDefaultSqContext()
{
    RdmaSqContextLite sq{};
    sq.qpn = 1;
    sq.sqVa = 0x10000;
    sq.wqeSize = 64;
    sq.depth = 128;
    sq.headAddr = 0x20000;
    sq.tailAddr = 0x20008;
    sq.sl = 0;
    sq.dbVa = 0x30000;
    sq.dbMode = 0;
    return sq;
}

static RdmaCqContextLite MakeDefaultCqContext()
{
    RdmaCqContextLite cq{};
    cq.cqn = 2;
    cq.cqVa = 0x40000;
    cq.cqeSize = 32;
    cq.cqDepth = 256;
    cq.headAddr = 0x50000;
    cq.tailAddr = 0x50008;
    cq.dbVa = 0x60000;
    cq.dbMode = 0;
    return cq;
}

class RdmaConnLiteV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RdmaConnLiteV2Test tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RdmaConnLiteV2Test tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in RdmaConnLiteV2Test SetUP" << std::endl;
        sqCtx_ = MakeDefaultSqContext();
        cqCtx_ = MakeDefaultCqContext();
        uniqueId_ = BuildRdmaConnLiteV2UniqueId(0, sqCtx_, cqCtx_);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RdmaConnLiteV2Test TearDown" << std::endl;
    }

    RdmaSqContextLite sqCtx_{};
    RdmaCqContextLite cqCtx_{};
    std::vector<char> uniqueId_;
};

TEST_F(RdmaConnLiteV2Test, Ut_When_Construct_Expect_Success)
{
    std::cout << "Start Ut_When_Construct_Expect_Success" << std::endl;
    
    RdmaConnLiteV2 connLite(uniqueId_);
    
    std::cout << "End Ut_When_Construct_Expect_Success" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;
    
    RdmaConnLiteV2 connLite(uniqueId_);
    
    std::string desc = connLite.Describe();
    EXPECT_FALSE(desc.empty());
    
    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_DmaMode_Expect_Correct)
{
    std::cout << "Start Ut_When_DmaMode_Expect_Correct" << std::endl;
    
    RdmaConnLiteV2 connLite(uniqueId_);
    
    EXPECT_EQ(connLite.dmaMode_, 0u);
    
    std::cout << "End Ut_When_DmaMode_Expect_Correct" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_SqContext_Expect_Valid)
{
    std::cout << "Start Ut_When_SqContext_Expect_Valid" << std::endl;
    
    RdmaConnLiteV2 connLite(uniqueId_);
    
    EXPECT_EQ(connLite.sqContext.qpn, sqCtx_.qpn);
    EXPECT_EQ(connLite.sqContext.sqVa, sqCtx_.sqVa);
    EXPECT_EQ(connLite.sqContext.wqeSize, sqCtx_.wqeSize);
    EXPECT_EQ(connLite.sqContext.depth, sqCtx_.depth);
    EXPECT_EQ(connLite.sqContext.headAddr, sqCtx_.headAddr);
    EXPECT_EQ(connLite.sqContext.tailAddr, sqCtx_.tailAddr);
    EXPECT_EQ(connLite.sqContext.sl, sqCtx_.sl);
    EXPECT_EQ(connLite.sqContext.dbVa, sqCtx_.dbVa);
    EXPECT_EQ(connLite.sqContext.dbMode, sqCtx_.dbMode);
    
    std::cout << "End Ut_When_SqContext_Expect_Valid" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_CqContext_Expect_Valid)
{
    std::cout << "Start Ut_When_CqContext_Expect_Valid" << std::endl;
    
    RdmaConnLiteV2 connLite(uniqueId_);
    
    EXPECT_EQ(connLite.cqContext.cqn, cqCtx_.cqn);
    EXPECT_EQ(connLite.cqContext.cqVa, cqCtx_.cqVa);
    EXPECT_EQ(connLite.cqContext.cqeSize, cqCtx_.cqeSize);
    EXPECT_EQ(connLite.cqContext.cqDepth, cqCtx_.cqDepth);
    EXPECT_EQ(connLite.cqContext.headAddr, cqCtx_.headAddr);
    EXPECT_EQ(connLite.cqContext.tailAddr, cqCtx_.tailAddr);
    EXPECT_EQ(connLite.cqContext.dbVa, cqCtx_.dbVa);
    EXPECT_EQ(connLite.cqContext.dbMode, cqCtx_.dbMode);
    
    std::cout << "End Ut_When_CqContext_Expect_Valid" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_DmaModeNonZero_Expect_Correct)
{
    std::cout << "Start Ut_When_DmaModeNonZero_Expect_Correct" << std::endl;
    
    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(1, sqCtx_, cqCtx_);
    RdmaConnLiteV2 connLite(testId);
    
    EXPECT_EQ(connLite.dmaMode_, 1u);
    
    std::cout << "End Ut_When_DmaModeNonZero_Expect_Correct" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_LargeValues_Expect_Correct)
{
    std::cout << "Start Ut_When_LargeValues_Expect_Correct" << std::endl;
    
    RdmaSqContextLite sqMax{};
    sqMax.qpn = UINT32_MAX;
    sqMax.sqVa = UINT64_MAX;
    sqMax.wqeSize = UINT32_MAX;
    sqMax.depth = UINT32_MAX;
    sqMax.headAddr = UINT64_MAX;
    sqMax.tailAddr = UINT64_MAX;
    sqMax.sl = UINT8_MAX;
    sqMax.dbVa = UINT64_MAX;
    sqMax.dbMode = INT8_MAX;

    RdmaCqContextLite cqMax{};
    cqMax.cqn = UINT32_MAX;
    cqMax.cqVa = UINT64_MAX;
    cqMax.cqeSize = UINT32_MAX;
    cqMax.cqDepth = UINT32_MAX;
    cqMax.headAddr = UINT64_MAX;
    cqMax.tailAddr = UINT64_MAX;
    cqMax.dbVa = UINT64_MAX;
    cqMax.dbMode = INT8_MAX;

    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(2, sqMax, cqMax);
    RdmaConnLiteV2 connLite(testId);
    
    EXPECT_EQ(connLite.dmaMode_, 2u);
    EXPECT_EQ(connLite.sqContext.qpn, UINT32_MAX);
    EXPECT_EQ(connLite.cqContext.cqn, UINT32_MAX);
    
    std::cout << "End Ut_When_LargeValues_Expect_Correct" << std::endl;
}


TEST_F(RdmaConnLiteV2Test, Ut_When_MultipleInstances_Expect_Independent)
{
    std::cout << "Start Ut_When_MultipleInstances_Expect_Independent" << std::endl;
    
    RdmaConnLiteV2 connLite1(uniqueId_);
    RdmaConnLiteV2 connLite2(uniqueId_);
    
    EXPECT_EQ(connLite1.dmaMode_, connLite2.dmaMode_);
    EXPECT_EQ(connLite1.sqContext.qpn, connLite2.sqContext.qpn);
    EXPECT_EQ(connLite1.cqContext.cqn, connLite2.cqContext.cqn);
    
    std::cout << "End Ut_When_MultipleInstances_Expect_Independent" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_DbModeSw_Expect_Correct)
{
    std::cout << "Start Ut_When_DbModeSw_Expect_Correct" << std::endl;
    
    RdmaSqContextLite sqSw = MakeDefaultSqContext();
    sqSw.dbMode = 1;
    RdmaCqContextLite cqSw = MakeDefaultCqContext();
    cqSw.dbMode = 1;
    
    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(0, sqSw, cqSw);
    RdmaConnLiteV2 connLite(testId);
    
    EXPECT_EQ(connLite.sqContext.dbMode, 1);
    EXPECT_EQ(connLite.cqContext.dbMode, 1);
    
    std::cout << "End Ut_When_DbModeSw_Expect_Correct" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_Write_SmallSize_Expect_SingleSlice)
{
    std::cout << "Start Ut_When_Write_SmallSize_Expect_SingleSlice" << std::endl;

    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(1, sqCtx_, cqCtx_);
    RdmaConnLiteV2 connLite(testId);

    // size < RDMA_DMA_MAX_SIZE，应只有 1 个分片(余数段)
    RmaBufSliceLite loc(0x1000, 4096, 0x11, 0);
    RmtRmaBufSliceLite rmt(0x2000, 4096, 0x22, 0, 0);
    u64 dbAddr = 0;
    u64 dbValue = 0;

    // 屏蔽对硬件 SQ 的真实读写
    MOCKER_CPP(&RdmaBaseOps::WaitSqFree).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::UpdateSqPI).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::CommitWqe).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_NO_THROW(connLite.Write(loc, rmt, dbAddr, dbValue));

    std::cout << "End Ut_When_Write_SmallSize_Expect_SingleSlice" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_Write_LargeSize_Expect_MultiSlice)
{
    std::cout << "Start Ut_When_Write_LargeSize_Expect_MultiSlice" << std::endl;

    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(1, sqCtx_, cqCtx_);
    RdmaConnLiteV2 connLite(testId);

    u64 totalSize = 0x80000000ULL + 0x1000ULL;
    RmaBufSliceLite loc(0x100000, totalSize, 0x11, 0);
    RmtRmaBufSliceLite rmt(0x200000, totalSize, 0x22, 0, 0);
    u64 dbAddr = 0;
    u64 dbValue = 0;

    // 屏蔽对硬件 SQ 的真实读写
    MOCKER_CPP(&RdmaBaseOps::WaitSqFree).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::UpdateSqPI).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::CommitWqe).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_NO_THROW(connLite.Write(loc, rmt, dbAddr, dbValue));

    // 校验上下文未被破坏（Describe 仍可正常输出）
    EXPECT_FALSE(connLite.Describe().empty());

    std::cout << "End Ut_When_Write_LargeSize_Expect_MultiSlice" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_WriteWithNotify_Expect_Success)
{
    std::cout << "Start Ut_When_WriteWithNotify_Expect_Success" << std::endl;

    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(1, sqCtx_, cqCtx_);
    RdmaConnLiteV2 connLite(testId);

    u64 totalSize = 0x80000000ULL;
    RmaBufSliceLite      loc(0x1000, totalSize, 0x11, 0);
    RmtRmaBufSliceLite   rmt(0x2000, totalSize, 0x22, 0, 0);

    // notify 地址段
    RmaBufSliceLite      locNotify(0x3000, 64, 0x33, 0);
    RmtRmaBufSliceLite   notify(0x4000, 64, 0x44, 0, 0);

    u64 dbAddr  = 0;
    u64 dbValue = 0;

    // 屏蔽对硬件 SQ 的真实读写
    MOCKER_CPP(&RdmaBaseOps::WaitSqFree).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::UpdateSqPI).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RdmaBaseOps::CommitWqe).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_NO_THROW(connLite.WriteWithNotify(loc, rmt, locNotify, notify, dbAddr, dbValue));

    std::cout << "End Ut_When_WriteWithNotify_Expect_Success" << std::endl;
}

TEST_F(RdmaConnLiteV2Test, Ut_When_GetVendorOpsCalledTwice_Expect_NoRecreate)
{
    std::cout << "Start Ut_When_GetVendorOpsCalledTwice_Expect_NoRecreate" << std::endl;

    std::vector<char> testId = BuildRdmaConnLiteV2UniqueId(1, sqCtx_, cqCtx_);
    RdmaConnLiteV2 connLite(testId);

    EXPECT_NE(connLite.rdmaOps_, nullptr);
    auto* firstPtr = connLite.rdmaOps_.get();

    // 再次显式调用 GetVendorOps，应直接早返回，不重复创建
    EXPECT_NO_THROW(connLite.GetVendorOps());
    EXPECT_EQ(connLite.rdmaOps_.get(), firstPtr);

    std::cout << "End Ut_When_GetVendorOpsCalledTwice_Expect_NoRecreate" << std::endl;
}