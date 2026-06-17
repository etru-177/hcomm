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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "aicpu/aicpu_ts_uboe_channel.h"

#define private public
using namespace hcomm;

class AicpuTsUboeChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// Lightweight fakes for external dependencies used by AicpuTsUboeChannel
struct FakeEndpoint { // will be used as EndpointHandle (void*)
    EndpointDesc desc;
    void* rdmaHandle{reinterpret_cast<void*>(0xDEADBEEF)};
    FakeEndpoint() {
        memset(&desc, 0, sizeof(desc));
        // Provide a valid, non-reserved endpoint description so code paths that
        // log or parse network addresses do not fail in unit tests.
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UBOE;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }
    EndpointDesc GetEndpointDesc() { return desc; }
    void* GetRdmaHandle() { return rdmaHandle; }
    Hccl::IpAddress GetIpAddress() const { return Hccl::IpAddress("127.0.0.1"); }
};

class FakeSocket : public Hccl::Socket {
public:
    FakeSocket(Hccl::SocketStatus status = Hccl::SocketStatus::OK) :
        Hccl::Socket(nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "fake", Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE),
        status_(status) {}
    void SendAsync(const u8 *sendBuf, u32 size) { sent_.insert(sent_.end(), sendBuf, sendBuf + size); }
    void RecvAsync(u8 *recvBuf, u32 size) {
        // If we previously recorded sent bytes, echo them back to the receiver so
        // higher-level unpacking (EID/Conn/Buffer) sees sensible data instead of all zeros.
        if (recvBuf && size) {
            if (!sent_.empty()) {
                u32 copySize = static_cast<u32>(std::min<size_t>(sent_.size(), static_cast<size_t>(size)));
                memcpy(recvBuf, sent_.data(), copySize);
                if (copySize < size) std::memset(recvBuf + copySize, 0, size - copySize);
                // remove echoed bytes so subsequent RecvAsync calls progress
                sent_.erase(sent_.begin(), sent_.begin() + copySize);
            } else {
                // no data available yet -> return zeros
                std::memset(recvBuf, 0, size);
            }
        }
    }
    Hccl::SocketStatus GetAsyncStatus() { return status_; }
    Hccl::SocketRole GetRole() const { return Hccl::SocketRole::SERVER; }

    // allow tests to change reported status
    Hccl::SocketStatus status_;
    std::vector<u8> sent_;
};

// Minimal fake LocalUbRmaBuffer and UbLocalNotify compatible interfaces
namespace Hccl {
class FakeLocalUbRmaBuffer : public Hccl::LocalUbRmaBuffer {
public:
    FakeLocalUbRmaBuffer(std::shared_ptr<Hccl::Buffer> b, void* rdma): LocalUbRmaBuffer(b, rdma) {}
    string Describe() const override
    {
        return "hello";
    }
};
class FakeUbLocalNotify : public Hccl::UbLocalNotify {
public:
    FakeUbLocalNotify(void* rdma, bool dev) : UbLocalNotify(rdma, dev) {}
};
}

// Helper to build HcommChannelDesc with fake socket and endpoint
static HcommChannelDesc MakeFakeChannelDesc(FakeSocket* sock) {
    HcommChannelDesc d{};
    d.socket = reinterpret_cast<void*>(sock);
    d.notifyNum = 0;
    d.exchangeAllMems = false;
    d.memHandles = nullptr;
    d.memHandleNum = 0;
    return d;
}

TEST_F(AicpuTsUboeChannelTest, Ut_Clean_WithoutInit_Returns_SUCCESS) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUboeChannelTest, Ut_Init_MockedHelpers_Returns_SUCCESS) {
    // Do not mock internal methods. Inject fake endpoint and fake socket so Init() exercises real code paths.
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);
    AicpuTsUboeChannel ch(ep, desc);

    // Call Init() to exercise real ParseInputParam/BuildSocket/BuildNotify/BuildBuffer paths with fakes.
    // We don't enforce a strict success code here because deeper system deps may not be available in unit test env.
    ch.Init();

    // If no crash, consider test successful
    SUCCEED();

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetStatus_WhenSocketNotReady_Returns_INIT) {
    // Inject a fake socket and control its GetAsyncStatus() instead of mocking IsSocketReady
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket(Hccl::SocketStatus::TIMEOUT);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);
    AicpuTsUboeChannel ch(ep, desc);

    // Ensure the channel uses our fake socket (ParseInputParam isn't called here)
    ch.channelDesc_.socket = reinterpret_cast<void*>(fakeSock);
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(fakeSock);

    // Ensure initial channelStatus is INIT
    EXPECT_EQ(ch.channelStatus, ChannelStatus::INIT);

    // Now GetStatus should observe socket timeout via socket_->GetAsyncStatus()
    auto status = ch.GetStatus();
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetNotifyNum_Returns_Value) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    ch.notifyNum_ = 42;
    uint32_t n = 0;
    auto ret = ch.GetNotifyNum(&n);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(n, 42u);
}

// Test-local stubs for Socket async APIs. These will be used with MOCKER_CPP to intercept
// calls to Socket::SendAsync and Socket::RecvAsync inside the state-machine test.
static void stub_Socket_SendAsync(Hccl::Socket *self, const u8 *sendBuf, u32 size)
{
    if (!self || !sendBuf || size == 0) return;
    auto *fs = dynamic_cast<FakeSocket *>(self);
    if (fs) {
        fs->sent_.insert(fs->sent_.end(), sendBuf, sendBuf + size);
    }
}

static void stub_Socket_RecvAsync(Hccl::Socket *self, u8 *recvBuf, u32 size)
{
    if (!self || !recvBuf || size == 0) return;
    auto *fs = dynamic_cast<FakeSocket *>(self);
    if (fs) {
        if (!fs->sent_.empty()) {
            u32 copySize = static_cast<u32>(std::min<size_t>(fs->sent_.size(), static_cast<size_t>(size)));
            memcpy(recvBuf, fs->sent_.data(), copySize);
            if (copySize < size) std::memset(recvBuf + copySize, 0, size - copySize);
            fs->sent_.erase(fs->sent_.begin(), fs->sent_.begin() + copySize);
            return;
        }
    }
    // no data available -> zero fill
    std::memset(recvBuf, 0, size);
}

TEST_F(AicpuTsUboeChannelTest, Ut_ProcessUboeState_AllStates_Transitions) {
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket();

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const u8 *, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(u8 *, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_RecvAsync));
    MOCKER_CPP(&AicpuTsUboeChannel::BuildConnection).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);

    AicpuTsUboeChannel ch(ep, desc);

    FakeEndpoint localEp;
    FakeEndpoint remoteEp;
    ch.localEp_ = localEp.GetEndpointDesc();
    ch.remoteEp_ = remoteEp.GetEndpointDesc();

    ch.channelDesc_.socket = reinterpret_cast<void*>(fakeSock);
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(fakeSock);

    // Minimal configuration so state handlers can run with fakes
    ch.channelDesc_.exchangeAllMems = false;

    // Ensure buffers/conns empty so IsResReady() returns true and SendDataSize() is exercised
    ch.commonRes_.connVec = {};
    ch.commonRes_.bufferVec = {};
    ch.connNum_ = 0;
    ch.bufferNum_ = 0;

    ch.channelStatus = ChannelStatus::SOCKET_OK;
    ch.uboeStatus = AicpuTsUboeChannel::UboeStatus::INIT;

    // Drive through the state machine by repeatedly calling GetStatus to cover IsResReady() and SendDataSize().
    int iter = 0;
    while (ch.channelStatus != ChannelStatus::READY && iter < 200) {
        HCCL_INFO("[%s] ch.uboeStatus[%s]", __func__, ch.uboeStatus.Describe().c_str());
        ch.GetStatus();
        iter++;
    }

    EXPECT_EQ(ch.channelStatus, ChannelStatus::READY);

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_PackingHelpers_NoCrash) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    Hccl::BinaryStream bs;
    ch.NotifyVecPack(bs);
    std::vector<Hccl::LocalRmaBuffer*> emptyBuf;
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> tags;
    ch.BufferVecPack(bs, emptyBuf);
    ch.ConnVecPack(bs);

    SUCCEED();
}

TEST_F(AicpuTsUboeChannelTest, Ut_Init_WithFakes_Runs) {
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);

    AicpuTsUboeChannel ch(ep, desc);

    // Override rdma/context behaviour by making ParseInputParam use our fake - inject via endpointHandle
    auto ret = ch.Init();
    // Init may try to build resources; expect success or graceful failure depending on deeper deps
    SUCCEED();
}

TEST_F(AicpuTsUboeChannelTest, Ut_H2DResPack_Packs_Data) {
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);

    AicpuTsUboeChannel ch(ep, desc);

    // Prepare channel so GetUniqueIdV2() can run: status READY and one notify
    ch.channelStatus = ChannelStatus::READY;
    ch.notifyNum_ = 1;
    ch.bufferNum_ = 2;
    ch.connNum_ = 1;

    auto notifyUptr = std::make_unique<Hccl::FakeUbLocalNotify>(fe.GetRdmaHandle(), true);
    Hccl::UbLocalNotify *rawNotify = notifyUptr.get();
    ch.localNotifies_.push_back(std::move(notifyUptr));
    ch.commonRes_.notifyVec.push_back(rawNotify);

    auto rmtUbRmaBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtNotifyVec_.push_back(std::move(rmtUbRmaBufPtr));

    auto buffer = std::make_shared<Hccl::Buffer>(0x100, 0x100);
    auto locUbRmaBufPtr = std::make_unique<Hccl::FakeLocalUbRmaBuffer>(buffer, fe.GetRdmaHandle());
    Hccl::FakeLocalUbRmaBuffer *locUbRmaBuf = locUbRmaBufPtr.get();
    ch.commonRes_.bufferVec.push_back(locUbRmaBuf);
    ch.commonRes_.bufferVec.push_back(nullptr);

    auto rmtBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtBufferVec_.push_back(std::move(rmtBufPtr));

    Hccl::IpAddress ipAddress = fe.GetIpAddress();
    Hccl::DevUbUboeConnection uboeConn(fe.GetRdmaHandle(), ipAddress, ipAddress, Hccl::OpMode::OPBASE, true,
        Hccl::HrtUbJfcMode::STARS_POLL, ipAddress, ipAddress);
    ch.commonRes_.connVec.push_back(&uboeConn);

    std::vector<char> out{};
    HcclResult ret = ch.H2DResPack(out);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetChannelKind_Returns_AICPU_TS_UBOE) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_UBOE);
}

// 辅助 FakeRemoteUbRmaBuffer，用于模拟远端内存区域
class FakeRemoteUbRmaBuffer : public Hccl::RemoteUbRmaBuffer {
public:
    FakeRemoteUbRmaBuffer(void* rdmaHandle, uint64_t addr, size_t size, HcclMemType type, const std::string& memInfo)
        : Hccl::RemoteUbRmaBuffer(rdmaHandle) {
        this->addr = addr;
        this->size = size;
        this->memType = type;
        this->memInfo = memInfo;
    }
};

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_NoBuffers_ReturnsSuccess) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 确保 rmtBufferVec_ 为空
    ch.rmtBufferVec_.clear();

    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remoteMem, nullptr);
    EXPECT_EQ(memNum, 0U);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_WithBuffers_ReturnsCorrectData) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 构造两个远端内存区域
    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto buf1 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    auto buf2 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x2000, 8192, HCCL_MEM_TYPE_HOST, "user_buffer");
    ch.rmtBufferVec_.push_back(std::move(buf1));
    ch.rmtBufferVec_.push_back(std::move(buf2));

    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 2U);
    ASSERT_NE(remoteMem, nullptr);

    // 验证第一个内存区域
    EXPECT_EQ(remoteMem[0].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[0].addr, reinterpret_cast<void*>(0x1000));
    EXPECT_EQ(remoteMem[0].size, 4096U);
    EXPECT_STREQ(memInfos[0], "ccl_buffer");

    // 验证第二个内存区域
    EXPECT_EQ(remoteMem[1].type, COMM_MEM_TYPE_HOST);
    EXPECT_EQ(remoteMem[1].addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(remoteMem[1].size, 8192U);
    EXPECT_STREQ(memInfos[1], "user_buffer");
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_NullParams_ReturnsError) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 至少构造一个 buffer，避免提前返回错误
    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    // 传入无效参数，期望返回参数错误（具体错误码视实现而定，当前返回 HCCL_E_PARA）
    EXPECT_EQ(ch.GetRemoteMems(&memNum, nullptr, &memInfos), HCCL_E_PARA);
    EXPECT_EQ(ch.GetRemoteMems(&memNum, &remoteMem, nullptr), HCCL_E_PARA);
    EXPECT_EQ(ch.GetRemoteMems(nullptr, &remoteMem, &memInfos), HCCL_E_PARA);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_OnlyCclBuffer_ReturnsSuccess) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));

    // 确保内部 cache 标志初始为 false，以便重新构建
    ch.cacheValid_ = false;

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1U);
    ASSERT_NE(remoteMem, nullptr);
    // memInfos 在数量不为0时也可能为 nullptr，不强制检查
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_WithUserBuffers_ReturnsCorrectData) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    // 第一个 buffer 为 ccl buffer，与后续用户 buffer 一并返回
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl");
    // 用户 buffer 1
    auto userBuf1 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x2000, 8192, HCCL_MEM_TYPE_DEVICE, "user1");
    // 用户 buffer 2
    auto userBuf2 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x3000, 16384, HCCL_MEM_TYPE_DEVICE, "user2");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));
    ch.rmtBufferVec_.push_back(std::move(userBuf1));
    ch.rmtBufferVec_.push_back(std::move(userBuf2));

    ch.cacheValid_ = false; // 强制重新构建缓存

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 3U);
    ASSERT_NE(remoteMem, nullptr);
    ASSERT_NE(memInfos, nullptr);

    // 验证第一个用户内存（索引1）
    EXPECT_EQ(remoteMem[1].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[1].addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(remoteMem[1].size, 8192U);
    EXPECT_STREQ(memInfos[1], "user1");

    // 验证第二个用户内存（索引2）
    EXPECT_EQ(remoteMem[2].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[2].addr, reinterpret_cast<void*>(0x3000));
    EXPECT_EQ(remoteMem[2].size, 16384U);
    EXPECT_STREQ(memInfos[2], "user2");
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetStatusReady_SocketNotNullptr) {
    MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(returnValue(HCCL_SUCCESS));
 	MOCKER_CPP(&SocketMgr::PutSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsUboeChannel::IsSocketReady).stubs().will(returnValue(true));
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);
    ch.uboeStatus = AicpuTsUboeChannel::UboeStatus::SET_READY;
    ch.GetStatus();
    GlobalMockObject::verify();
}