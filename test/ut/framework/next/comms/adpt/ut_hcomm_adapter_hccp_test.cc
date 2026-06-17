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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_adapter_hccp.h"

using namespace hcomm;

class HcommAdapterHccpTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HcommAdapterHccpTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HcommAdapterHccpTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcommAdapterHccpTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcommAdapterHccpTest TearDown" << std::endl;
    }
};

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_VersionEnough_Expect_Success)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_VersionNotEnough_Expect_NotSupport)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION - 1;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));
    
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_RaGetInterfaceFailed_Expect_E_INTERNAL)
{
    u32 devPhyId = 1;
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitSet_Expect_True)
{
    u32 devFeature = 1 << UBOE_DEV_FLAG_RIGHT_SHIFT;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
    
    // 测试其他位也有值的情况
    devFeature = (1 << UBOE_DEV_FLAG_RIGHT_SHIFT) | 0xFFFF;
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitNotSet_Expect_False)
{
    u32 devFeature = 0;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
    
    // 测试只有其他位被设置，但UBOE位未设置
    devFeature = 0xFFFFFFFF & ~(1 << UBOE_DEV_FLAG_RIGHT_SHIFT);
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
}

