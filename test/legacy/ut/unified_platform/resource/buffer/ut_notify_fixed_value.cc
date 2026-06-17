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
#include "notify_fixed_value.h"
#include "dev_capability.h"
#include "orion_adapter_rts.h"

using namespace Hccl;

class NotifyFixedValueTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NotifyFixedValueTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NotifyFixedValueTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in NotifyFixedValueTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in NotifyFixedValueTest TearDown" << std::endl;
    }
};

TEST_F(NotifyFixedValueTest, notify_fixed_value_get_addr_and_size)
{
    // Given
    MOCKER(HrtGetDeviceType).
        stubs().
        will(returnValue((DevType)DevType::DEV_TYPE_950));
 
    void* fakeAddr = nullptr;
    MOCKER(HrtMalloc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeAddr));
 
    NotifyFixedValue notifyFixedValue;
    // when
    u64 addrRes = notifyFixedValue.GetAddr();
    u32 sizeRes = notifyFixedValue.GetSize();

    // then
    EXPECT_EQ(0, addrRes);
    EXPECT_EQ(8, sizeRes);
}