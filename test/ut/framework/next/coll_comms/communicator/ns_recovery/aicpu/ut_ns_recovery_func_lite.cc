#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "ns_recovery/aicpu/ns_recovery_func_lite.h"

using namespace hccl;

class NsRecoveryFuncLiteTest : public testing::Test {
protected:
    void SetUp() override { GlobalMockObject::reset(); }
    void TearDown() override { GlobalMockObject::reset(); }

    static void MockHalTsdrvCtlReturn(int rv)
    {
        MOCKER(halTsdrvCtl).stubs().with(mockcpp::any()).will(returnValue(rv));
    }
};

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenDrvErrorExpectHcclEDrv)
{
    // 模拟 halTsdrvCtl 返回驱动不支持
    MockHalTsdrvCtlReturn(DRV_ERROR_NOT_SUPPORT);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_DRV);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenSuccessExpectHcclSuccess)
{
    // 模拟 halTsdrvCtl 返回成功，step为0时会立即认为满足条件
    MockHalTsdrvCtlReturn(DRV_ERROR_NONE);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenTimeoutExpectHcclETimeout)
{
    // 模拟 halTsdrvCtl 返回成功，但返回的status保持小于step，设置超时时间触发超时分支
    MockHalTsdrvCtlReturn(DRV_ERROR_NONE);
    // 使用step=1，timeout设为1微秒（远小于函数内部每次循环会睡眠的5毫秒），以便快速触发超时分支
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 1, 1000ULL);
    EXPECT_EQ(ret, HcclResult::HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}