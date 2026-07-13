/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * This SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "channel_process.h"
#include "aiv_urma_channel.h"
#include "aicpu_ts_roce_channel_v2.h"

class TestChannelProcess : public TestHcommCAdptBase {
public:
    void SetUp() override {
        TestHcommCAdptBase::SetUp();
    }
    void TearDown() override {
        {
            std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
            hcomm::ChannelProcess::g_ChannelMap.clear();
            hcomm::ChannelProcess::g_ChannelD2HMap.clear();
        }
        TestHcommCAdptBase::TearDown();
    }
};

// CreateChannelsLoop 空指针测试
TEST_F(TestChannelProcess, Ut_TestCreateChannelsLoop_When_EndpointHandleNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc desc = {};
    ChannelHandle outHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::CreateChannelsLoop(nullptr, COMM_ENGINE_AICPU, &desc, 1, outHandles);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ConnectChannels 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestConnectChannels_When_TargetChannelsNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = hcomm::ChannelProcess::ConnectChannels(nullptr, 1, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestConnectChannels_When_ChannelNumZero_Return_HCCL_E_PARA)
{
    ChannelHandle channels[1] = {};
    HcclResult ret = hcomm::ChannelProcess::ConnectChannels(channels, 0, COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// FillChannelD2HMap 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_DeviceChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(nullptr, hostHandles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_ListNumZero_Return_HCCL_E_PARA)
{
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, hostHandles, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// LaunchChannelKernelCommon 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        nullptr, hostHandles, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, nullptr, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ListNumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, hostHandles, hcommDescs, 0, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// SaveChannels 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestSaveChannels_When_TargetChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle userChannels[1] = {};
    HcclResult ret = hcomm::ChannelProcess::SaveChannels(
        nullptr, userChannels, hcommDescs, 1, COMM_ENGINE_AICPU, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestSaveChannels_When_UserChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle targetChannels[1] = {};
    HcclResult ret = hcomm::ChannelProcess::SaveChannels(
        targetChannels, nullptr, hcommDescs, 1, COMM_ENGINE_AICPU, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestSaveChannels_When_ChannelNumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle targetChannels[1] = {};
    ChannelHandle userChannels[1] = {};
    HcclResult ret = hcomm::ChannelProcess::SaveChannels(
        targetChannels, userChannels, hcommDescs, 0, COMM_ENGINE_AICPU, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ChannelGetRemoteMems 空指针测试
TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_RemoteMemNullptr_Return_HCCL_E_PTR)
{
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, nullptr, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemInfosNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemNumNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, nullptr, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_ChannelClean_NullList_Returns_E_PARA) {
    // Passing null pointer should return parameter error
    auto ret = hcomm::ChannelProcess::ChannelClean(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_ChannelResumeConcurrency_ZeroChannels_Returns_SUCCESS) {
    // Zero channel count should be a no-op and return success
    ChannelHandle dummyList[1] = {0};
    auto ret = hcomm::ChannelProcess::ChannelResumeConcurrency(dummyList, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_When_ResumeConcurrencyFails_ReturnsError) {
    ChannelHandle list[1] = { (ChannelHandle)0x1 };
    // Mock ChannelResumeConcurrency to return internal error
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelResumeConcurrency, HcclResult(const ChannelHandle*, uint32_t))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto ret = hcomm::ChannelProcess::ChannelResume(list, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_NullList_Returns_E_PARA) {
    auto ret = hcomm::ChannelProcess::ChannelResume(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_LaunchChannelKernel_When_ChannelKindIsUBOE_CallsChannelKernelLaunchForBase) {
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};

    class FakeUboeChannel : public hcomm::Channel {
    public:
        hcomm::HcommChannelKind GetChannelKind() const override {
            return hcomm::HcommChannelKind::AICPU_TS_UBOE;
        }
        HcclResult Init() override {
            return HCCL_SUCCESS;
        }
        HcclResult GetNotifyNum(uint32_t *notifyNum) const override {
            return HCCL_SUCCESS;
        }
        HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override {
            return HCCL_SUCCESS;
        }
        hcomm::ChannelStatus GetStatus() override {
            return hcomm::ChannelStatus::READY;
        }
        HcclResult Clean() override {
            return HCCL_SUCCESS;
        }
        HcclResult Resume() override {
            return HCCL_SUCCESS;
        }
        HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override {
            return HCCL_SUCCESS;
        }
        HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override {
            return HCCL_SUCCESS;
        }
        HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override {
            return HCCL_SUCCESS;
        }
        HcclResult Write(void *dst, const void *src, uint64_t len) override {
            return HCCL_SUCCESS;
        }
        HcclResult Read(void *dst, const void *src, uint64_t len) override {
            return HCCL_SUCCESS;
        }
        HcclResult ChannelFence() override {
            return HCCL_SUCCESS;
        }
    };

    FakeUboeChannel fakeChannel;
    ChannelHandle hostHandles[1] = {reinterpret_cast<ChannelHandle>(&fakeChannel)};

    MOCKER_CPP(&hcomm::ChannelProcess::LaunchChannelKernelCommon,
        HcclResult(ChannelHandle*, ChannelHandle*, HcommChannelDesc*, uint32_t, const std::string&, aclrtBinHandle, const std::string&, bool))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = hcomm::ChannelProcess::LaunchChannelKernel(deviceHandles, hostHandles, hcommDescs, 1, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult StubAivBuildChannelEntityToDevice(hcomm::AivUrmaChannel *channel, void **devPtr)
{
    (void)channel;
    if (devPtr == nullptr) {
        return HCCL_E_PTR;
    }
    *devPtr = reinterpret_cast<void *>(0x5678);
    return HCCL_SUCCESS;
}

HcclResult StubRoceBuildAndGetDevChannelEntity(hcomm::AicpuTsRoceChannelV2 *channel, uint64_t* devChannelEntityPtr)
{
    (void)channel;
    if (devChannelEntityPtr == nullptr) {
        return HCCL_E_PTR;
    }
    *devChannelEntityPtr = 0x1234;
    return HCCL_SUCCESS;
}

// SaveAivChannels UBC_CTP 协议测试
TEST_F(TestChannelProcess, Ut_SaveAivChannels_When_UbcCtp_Expect_BuildDevEntity)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_CTP;
    hcomm::AivUrmaChannel aivUrmaChannel(endpointHandle, channelDesc);

    ChannelHandle targetChannels[1] = {reinterpret_cast<ChannelHandle>(&aivUrmaChannel)};
    ChannelHandle userChannels[1] = {0};
    HcommChannelDesc channelDescs[1] = {channelDesc};

    MOCKER_CPP(&hcomm::AivUrmaChannel::BuildChannelEntityToDevice, HcclResult(hcomm::AivUrmaChannel::*)(void **))
        .stubs()
        .will(invoke(StubAivBuildChannelEntityToDevice));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::SaveAivChannels(targetChannels, userChannels, channelDescs, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(userChannels[0], static_cast<ChannelHandle>(0x5678));
}

// SaveAivChannels ROCE 协议测试
TEST_F(TestChannelProcess, Ut_SaveAivChannels_When_Roce_Expect_BuildDevEntity)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    hcomm::AicpuTsRoceChannelV2 roceChannel(endpointHandle, channelDesc, COMM_ENGINE_AIV);

    ChannelHandle targetChannels[1] = {reinterpret_cast<ChannelHandle>(&roceChannel)};
    ChannelHandle userChannels[1] = {0};
    HcommChannelDesc channelDescs[1] = {channelDesc};

    MOCKER_CPP(&hcomm::AicpuTsRoceChannelV2::BuildAndGetDevChannelEntity, HcclResult(hcomm::AicpuTsRoceChannelV2::*)(uint64_t*))
        .stubs()
        .will(invoke(StubRoceBuildAndGetDevChannelEntity));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::SaveAivChannels(targetChannels, userChannels, channelDescs, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(userChannels[0], static_cast<ChannelHandle>(0x1234));
}

// SaveAivChannels 不支持的协议测试
TEST_F(TestChannelProcess, Ut_SaveAivChannels_When_ProtocolUnsupported_Expect_SUCCESS)
{
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_RESERVED;

    ChannelHandle targetChannels[1] = {0};
    ChannelHandle userChannels[1] = {0};
    HcommChannelDesc channelDescs[1] = {channelDesc};

    HcclResult ret = hcomm::ChannelProcess::SaveAivChannels(targetChannels, userChannels, channelDescs, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 可配置 GetStatus 返回值的 Fake Channel，用于 ChannelGetStatus 测试
class FakeStatusChannel : public hcomm::Channel {
public:
    explicit FakeStatusChannel(hcomm::ChannelStatus status) : status_(status)
    {
    }
    hcomm::HcommChannelKind GetChannelKind() const override
    {
        return hcomm::HcommChannelKind::AICPU_TS_UBOE;
    }
    HcclResult Init() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override
    {
        return HCCL_SUCCESS;
    }
    HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override
    {
        return HCCL_SUCCESS;
    }
    hcomm::ChannelStatus GetStatus() override
    {
        return status_;
    }
    HcclResult Clean() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Resume() override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult WriteWithNotify(void *dst, const void *src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Read(void *dst, const void *src, uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ChannelFence() override
    {
        return HCCL_SUCCESS;
    }

private:
    hcomm::ChannelStatus status_;
};

// 将 channel 注册到 g_ChannelMap / g_ChannelD2HMap（deviceId=0，handle 即 channel 指针值）
static ChannelHandle RegisterFakeChannel(std::shared_ptr<hcomm::Channel> channel)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(channel.get());
    hcomm::ChannelProcess::g_ChannelMap[handle] = std::move(channel);
    hcomm::DeviceChannelKey key{0, handle};
    hcomm::ChannelProcess::g_ChannelD2HMap[key] = handle;
    return handle;
}

// ChannelGetStatus channelList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_ChannelListNullptr_Return_HCCL_E_PTR)
{
    int32_t statusList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(nullptr, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ChannelGetStatus statusList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channelList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 全部通道 READY 返回 SUCCESS，并回写状态
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_AllReady_Return_SUCCESS)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// 存在 FAILED 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasFailed_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
}

// 存在 SOCKET_TIMEOUT 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasSocketTimeout_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::SOCKET_TIMEOUT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::SOCKET_TIMEOUT));
}

// 存在未到终态通道（INIT）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasNotReady_Return_HCCL_E_AGAIN)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// READY + FAILED 混合（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndFailed_Return_HCCL_E_NETWORK)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// READY + INIT 混合（存在未到终态通道）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndInit_Return_HCCL_E_AGAIN)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chInit = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hInit = RegisterFakeChannel(chInit);
    ChannelHandle channelList[2] = {hReady, hInit};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// statusList 已预标记 FAILED 的通道被 continue 跳过，不再获取状态，但计入 failCount
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListPreMarkedFailed_SkipGetStatus)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    // 第二个通道预标记为 FAILED，触发 continue 分支，不再调用其 GetStatus
    int32_t statusList[2] = {0, static_cast<int32_t>(hcomm::ChannelStatus::FAILED)};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    // 被跳过的通道计入 failCount，故 readyCount(1)+failCount(1)=2 == listNum，且非全 READY，返回 HCCL_E_NETWORK
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    // 被跳过通道的状态保持预标记值不变
    EXPECT_EQ(statusList[1], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
    // 第一个通道状态被刷新为 READY
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// handle 未注册时 WithChannelByHandleLocked 返回 HCCL_E_NOT_FOUND
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HandleNotFound_Return_HCCL_E_NOT_FOUND)
{
    ChannelHandle channelList[1] = {static_cast<ChannelHandle>(0xDEADBEEFULL)};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
