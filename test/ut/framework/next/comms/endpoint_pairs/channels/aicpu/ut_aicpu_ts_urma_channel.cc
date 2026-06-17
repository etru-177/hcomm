#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_urma_channel.h"
#undef private
#undef protected
using namespace hcomm;

class AicpuTsUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsUrmaChannelTest, Ut_Clean_WithoutInit_Returns_SUCCESS) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_Resume_MockedBuilds_Returns_SUCCESS) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // Mock private helper methods BuildConnection and BuildUbMemTransport
    MOCKER_CPP(&AicpuTsUrmaChannel::BuildSocket, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::BuildConnection, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::BuildUbMemTransport, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_GetStatus_DfxInfo_TEST) {
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // 1. 构造依赖
    Hccl::BaseMemTransport::Attribution attr{};
    Hccl::IpAddress locAddrIp{"1.1.1.1"};
    Hccl::IpAddress rmtAddrIp{"2.2.2.2"};
    Hccl::Socket sock{
        nullptr,
        locAddrIp,
        0,
        rmtAddrIp,
        "ut",
        Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE};
    Hccl::BaseMemTransport::CommonLocRes commonLocRes{};
    std::vector<Hccl::RmaConnection *> conns;
    auto conn = std::make_unique<Hccl::DevUbConnection>(nullptr, locAddrIp, rmtAddrIp, Hccl::OpMode::OPBASE);
    conns.push_back(conn.get()); // 仅占位，测试不涉及具体连接对象
    commonLocRes.connVec = conns;
    RdmaHandle rdmaHandle = nullptr;
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};

    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::HOST_NET;
    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::UB_TP;
    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    uint32_t locDevPhyId = 0;
    uint32_t rmtDevPhyId = 0;
    Hccl::LinkData ld = Hccl::LinkData(
        portDeploymentType,
        linkProtocol, 
        locDevPhyId, rmtDevPhyId,
        locAddr, rmtAddr
    );

    auto memTransport = std::make_unique<Hccl::UbMemTransport>(commonLocRes, attr, ld, sock, rdmaHandle, locCntNotifyRes, false);
    memTransport->connNum = 1;
    memTransport->baseStatus = Hccl::TransportStatus::READY;
    ch.memTransport_ = std::move(memTransport);

    MOCKER_CPP(&Hccl::UbMemTransport::Describe, HcclResult (Hccl::UbMemTransport::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_PARA)); // 连续两次调用，第一次成功，第二次失败

    auto ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);

    // 再次调用GetStatus，验证isFirstPrintChannelInfo_逻辑分支覆盖
    ch.isFirstPrintChannelInfo_ = true; // 重置为true，模拟第一次打印
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

TEST_F(AicpuTsUrmaChannelTest, Ut_StartListen_When_RoleNotServer_Expect_SUCCESS)
{
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.StartListen();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
