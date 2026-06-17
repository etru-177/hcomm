/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>

#define private public
#define protected public
#include "ccu_res_batch_allocator.h"

#include "hccl_common_v2.h"
#include "ccu_component.h"

#undef private
#undef protected

using namespace Hccl;

/*
 * 重要事项请注意：
 * 因单例不容易实现完全打桩，ccu_component用例运行后
 * 部分单例已在内存中，故以下用例调整需注意设备号
 */
extern void MockCcuResources(const int32_t devLogicId, const CcuVersion ccuVersion);
extern void MockCcuNetworkDevice(const int32_t devLogicId);

class CcuResBatchAllocatorTest: public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResBatchAllocatorTest tests set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResBatchAllocatorTest tests tear down." << std::endl;
    }
 
    virtual void SetUp()
    {
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResBatchAllocatorTest SetUP" << std::endl;
    }
 
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResBatchAllocatorTest TearDown" << std::endl;
    }
};

void CheckRes(CcuResRepository &ccuResRepo)
{
    std::cout << "------------------------" << std::endl;
    for (int i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        std::cout << "BlockMS: ";
        auto blockMsInfos = ccuResRepo.blockMs[i];
        for (int j = 0; j < blockMsInfos.size(); j++) {
            std::cout << blockMsInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "BlockLoop: ";
        auto blockLoopInfos = ccuResRepo.blockLoopEngine[i];
        for (int j = 0; j < blockLoopInfos.size(); j++) {
            std::cout << blockLoopInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "BlockCke: ";
        auto blockCkeInfos = ccuResRepo.blockCke[i];
        for (int j = 0; j < blockCkeInfos.size(); j++) {
            std::cout << blockCkeInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "MS: ";
        auto msInfos = ccuResRepo.ms[i];
        for (int j = 0; j < msInfos.size(); j++) {
            std::cout << msInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Loop: ";
        auto loopInfos = ccuResRepo.loopEngine[i];
        for (int j = 0; j < loopInfos.size(); j++) {
            std::cout << loopInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Cke: ";
        auto ckeInfos = ccuResRepo.cke[i];
        for (int j = 0; j < ckeInfos.size(); j++) {
            std::cout << ckeInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Xn: ";
        auto xnInfos = ccuResRepo.xn[i];
        for (int j = 0; j < xnInfos.size(); j++) {
            std::cout << xnInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Gsa: ";
        auto gsaInfos = ccuResRepo.gsa[i];
        for (int j = 0; j < gsaInfos.size(); j++) {
            std::cout << gsaInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;

        std::cout << "Mission: ReqType: " << (int)ccuResRepo.mission.reqType << " ";
        auto missionInfos = ccuResRepo.mission.mission[i];
        for (int j = 0; j < missionInfos.size(); j++) {
            std::cout << missionInfos[j].Describe() << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------" << std::endl;
}

void DumpBlockResInfo(ResType resType, const std::vector<BlockInfo> &blocks)
{
    HCCL_INFO("Dump ResType[%s] block resources info: ", resType.Describe().c_str());
    uint32_t blockNum = blocks.size();
    for (size_t k = 0; k < blockNum; k++) {
        HCCL_INFO("Block[id[%u], startId[%u], num[%u], handle(uintptr_t)[%llu], allocated[%d]]",
            blocks[k].id, blocks[k].startId, blocks[k].num, blocks[k].handle,
            static_cast<int>(blocks[k].allocated));
    }
}

void MockerCcuComponent(const int32_t devLogicId, const CcuVersion ccuVersion)
{
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);
    EXPECT_NO_THROW(CcuComponent::GetInstance(devLogicId).Init());
}

TEST_F(CcuResBatchAllocatorTest, Ut_Init_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());
}

TEST_F(CcuResBatchAllocatorTest, Ut_Init_When_AX_Mainboard_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 3; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);
    auto &ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    ccuResSpecs.isAX = true;

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());
    constexpr uint32_t BLOCK_SIZE_MS_AX_DIE0 = 128;
    EXPECT_EQ(allocater.resStrategys[0].msNum, BLOCK_SIZE_MS_AX_DIE0);
    ASSERT_EQ(allocater.resBlocks[0].size(), 3);
    DumpBlockResInfo(ResType::LOOP, allocater.resBlocks[0][0]);
    DumpBlockResInfo(ResType::MS, allocater.resBlocks[0][1]);
    DumpBlockResInfo(ResType::CKE, allocater.resBlocks[0][2]);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockLoopEngineReq[0] = 1;
    resReq.loopEngineReq[0] = 2;
    
    resReq.blockCkeReq[0] = 65;
    resReq.ckeReq[0] = 128;
    
    resReq.blockMsReq[0] = 512;

    resReq.gsaReq[0] = 1024;

    resReq.missionReq.req[0] = {3};

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    CheckRes(ccuResRepo);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumIsEmpty_Expect_Return_ErrorPara)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq; // 检查空申请
    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumIsMaxNum_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockCkeReq[1] = 0;
    resReq.blockLoopEngineReq[0] = 192;
    resReq.loopEngineReq[0] = 8; // {8, 8};
    resReq.blockMsReq[0] = 1536; // {1536, 1536};
    resReq.blockCkeReq[0] = 128; // {128, 128};
    resReq.ckeReq[0] = 832; // {832, 832};
    resReq.gsaReq[0] = 3072; // {3072, 3072};
    resReq.xnReq[0] = 3072; // {3072, 3072};
    resReq.missionReq.req[0] = 16; // {16, 16};

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    CheckRes(ccuResRepo);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocResHandle_When_CcuV1AndResNumExceedsLeftNum_Expect_Return_ErrorUnavalible)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResReq resReq;
    resReq.blockLoopEngineReq[0] = 1; // {1, 0};
    resReq.loopEngineReq[0] = 2; // {2, 3};

    resReq.blockLoopEngineReq[0] = 1;
    resReq.loopEngineReq[0] = 2;
    resReq.missionReq.req[0] = 2;
    
    resReq.blockCkeReq[0] = 65; // {65, 0};
    resReq.ckeReq[0] = 129; // {129, 0};
    
    // 1. 资源申请超过了一半，故第二次申请资源会不足
    resReq.blockMsReq[0] = 64 * 13; // {64 * 13, 0};

    resReq.missionReq.req[0] = 3; // {3, 3};
    resReq.missionReq.req[1] = 2; // 会选用较多的，即 3

    CcuResHandle handle;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    CcuResHandle errorHandle;
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 2. 申请超过剩余资源的loop
    resReq = {}; // 重置错误的请求
    resReq.loopEngineReq[0] = 50; // 申请超过剩余资源
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 3. 申请超过mission规格的mission
    resReq = {}; // 重置错误的请求
    resReq.missionReq.req[0] = 17;
    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    // 释放其他资源避免影响其他用例
    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 4. mission资源申请超过一半，故第二次申请资源会不足
    resReq = {};
    resReq.missionReq.req[0] = 9;
    ret = allocater.AllocResHandle(resReq, handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    ret = allocater.AllocResHandle(resReq, errorHandle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(errorHandle, nullptr);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetResourceAndReleaseResHandle_When_resHandleIsInvalid_Expect_Return_ErrorPara)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockerCcuComponent(devLogicId, ccuVersion);

    CcuResBatchAllocator &allocater = CcuResBatchAllocator::GetInstance(devLogicId);
    allocater.devLogicId = devLogicId;

    EXPECT_NO_THROW(allocater.Init());

    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuResHandle handle = nullptr;
    CcuResRepository ccuResRepo;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    handle = (CcuResHandle)0x89674878;
    ret = allocater.GetResource(handle, ccuResRepo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = allocater.ReleaseResHandle(handle);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocConsecutiveRes_When_AllocRes_fail_Expect_HCCL_E_PARA)
{
    // 前置条件
    MOCKER_CPP(&CcuComponent::AllocRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_E_PARA));
    CcuResReq resReq;
    resReq.continuousXnReq[0] = 1;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    resRepoPtr->continuousXn[0].push_back(ResInfo(5, 3));
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.AllocConsecutiveRes(resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_TryAllocResHandle_When_AllocContinuousRes_fail_Expect_HCCL_E_PARA)
{
    // 前置条件
    MOCKER_CPP(&CcuResBatchAllocator::AllocBlockRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuResBatchAllocator::CcuMissionMgr::Alloc).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::AllocRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_E_PARA));
    std::unique_ptr<CcuResRepository> resRepoPtr = std::make_unique<CcuResRepository>();
    uintptr_t handleKey  = reinterpret_cast<uintptr_t>(resRepoPtr.get());
    CcuResReq resReq;
    resReq.continuousXnReq[0] = 1;
    resRepoPtr->continuousXn[0].push_back(ResInfo(5, 3));
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.TryAllocResHandle(handleKey, resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CcuResBatchAllocatorTest, Ut_AllocContinuousRes_When_AllocRes_success_Expect_HCCL_SUCCESS)
{
    // 前置条件
    MOCKER_CPP(&CcuComponent::AllocRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    CcuResReq resReq;
    auto resRepoPtr = std::make_unique<CcuResRepository>();
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.AllocConsecutiveRes(resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_TryAllocResHandle_When_AllocContinuousRes_success_Expect_HCCL_SUCCESS)
{
    // 前置条件
    MOCKER_CPP(&CcuResBatchAllocator::AllocBlockRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuResBatchAllocator::CcuMissionMgr::Alloc).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::AllocRes).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<CcuResRepository> resRepoPtr = std::make_unique<CcuResRepository>();
    uintptr_t handleKey  = reinterpret_cast<uintptr_t>(resRepoPtr.get());
    CcuResReq resReq;
    CcuResBatchAllocator ccuResBatchAllocator;
    ccuResBatchAllocator.dieEnableFlags[0] = true;
    ccuResBatchAllocator.dieEnableFlags[1] = false;

    // 执行步骤
    auto ret = ccuResBatchAllocator.TryAllocResHandle(handleKey, resReq, resRepoPtr);

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}