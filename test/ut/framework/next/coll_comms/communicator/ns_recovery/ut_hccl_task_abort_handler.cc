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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#include "coll_comm.h"
#include "ns_recovery/task_abort_handler.h"
#include "ccu_device_pub.h"
#undef private

using namespace hccl; 
using namespace hcomm;

class HcclTaskAbortHandlerTest : public testing::Test {
public:
    static void SetUpTestCase()
    {
        std::cout << "HcclTaskAbortHandlerTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HcclTaskAbortHandlerTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        HcclTaskAbortHandler::GetInstance().commVector_.clear();
        std::cout << "A Test case in HcclTaskAbortHandlerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcclTaskAbortHandlerTest TearDown" << std::endl;
        HcclTaskAbortHandler::GetInstance().commVector_.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_success)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm *comm = nullptr;
    HcclTaskAbortHandler::GetInstance().Register(comm);
    void* args = reinterpret_cast<void*>(&HcclTaskAbortHandler::GetInstance().commVector_);

    // 模拟 Suspend 方法返回成功
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    HcclTaskAbortHandler::GetInstance().UnRegister(comm);
    std::cout << "test_task_abort_handle_call_back_stage_pre_success completed" << std::endl;
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_fail)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm *comm = nullptr;
    HcclTaskAbortHandler::GetInstance().Register(comm);
    void* args = reinterpret_cast<void*>(&HcclTaskAbortHandler::GetInstance().commVector_);

    // 模拟 Suspend 方法返回失败
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 清理
    HcclTaskAbortHandler::GetInstance().UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_success)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm *comm = nullptr;
    HcclTaskAbortHandler::GetInstance().Register(comm);
    void* args = reinterpret_cast<void*>(&HcclTaskAbortHandler::GetInstance().commVector_);

    // 模拟 CCU 相关函数返回成功
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));

    // 模拟 Clean 方法返回成功
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    HcclTaskAbortHandler::GetInstance().UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_fail)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm *comm = nullptr;
    HcclTaskAbortHandler::GetInstance().Register(comm);
    void* args = reinterpret_cast<void*>(&HcclTaskAbortHandler::GetInstance().commVector_);

    // 模拟 CCU 相关函数返回成功
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));

    // 模拟 Clean 方法返回失败
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm:: *)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 清理
    HcclTaskAbortHandler::GetInstance().UnRegister(comm);
}
