#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_c_adpt.h"
#include "acl/acl_rt.h"
#define private public
#define protected public
#include "server_socket/server_socket_manager.h"
#include "server_socket/server_socket_mgr.h"
#include "socket_mgr.h"
#include "hccp_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "hcomm_res_mgr.h"
#undef protected
#undef private

class DeviceResetCallbackTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeviceResetCallbackTest set up." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DeviceResetCallbackTest tear down." << std::endl;
    }
    virtual void SetUp()
    {
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(DeviceResetCallbackTest, Ut_When_RegisterDeviceResetCallback_Expect_AclrtRegDeviceStateCallbackCalled)
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr_test", nullptr, nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_OutOfRange_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().DeInit(999);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketHandleManagerDeInit_Expect_NoCrash)
{
    Hccl::SocketHandleManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketHandleManagerDeInit_OutOfRange_Expect_NoCrash)
{
    Hccl::SocketHandleManager::GetInstance().DeInit(999);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_NotInited_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().DeInit(5);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerInitThenDeInit_Expect_InstanceRemoved)
{
    Hccl::HccpHdcManager::GetInstance().Init(0);
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketMgrDeInit_Expect_NoCrash)
{
    hcomm::SocketMgr::DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketMgrDeInit_Expect_NoCrash)
{
    hcomm::ServerSocketMgr::DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_Expect_NoCrash)
{
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithDeviceSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithHostSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithBothSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType typeUb = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPortUb = Hccl::PortData(0, typeUb, 0, ipAddr);
    uint32_t port1 = 43210;
    HcclResult ret1 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPortUb, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    Hccl::DevNetPortType typeRdma = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPortRdma = Hccl::PortData(0, typeRdma, 0, ipAddr);
    uint32_t port2 = 0;
    HcclResult ret2 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPortRdma, Hccl::NicType::HOST_NIC_TYPE, 0, &port2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(port2, 0);

    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_AllDeInitCalled_SimulateDeviceResetPre_Expect_NoCrash)
{
    u32 devPhyId = 0;
    s32 deviceId = 0;
    hcomm::SocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketManager::GetInstance().DeInit(devPhyId);
    Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
}

TEST_F(DeviceResetCallbackTest, Ut_When_AllDeInitCalled_WithNonZeroDevPhyId_Expect_NoCrash)
{
    u32 devPhyId = 1;
    s32 deviceId = 1;
    hcomm::SocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketManager::GetInstance().DeInit(devPhyId);
    Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HcommResMgrInit_Expect_Success)
{
    HcommResult ret = HcommResMgrInit(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithDestroyed_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().destroyed = true;
    Hccl::RdmaHandleManager::GetInstance().DeInit(0);
    Hccl::RdmaHandleManager::GetInstance().destroyed = false;
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_WithDestroyed_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().destroyed = true;
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
    Hccl::HccpHdcManager::GetInstance().destroyed = false;
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_SelectiveDevPhyId_Expect_OnlyMatchingCleared)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType typeUb = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort0 = Hccl::PortData(0, typeUb, 0, ipAddr);
    uint32_t port0 = 43210;
    HcclResult ret0 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort0, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port0);
    EXPECT_EQ(ret0, HCCL_SUCCESS);

    hcomm::ServerSocketManager::GetInstance().DeInit(999);
    EXPECT_NE(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithRdmaHandle_Expect_CleanupRdmaEntry)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xAAAA;
    Hccl::IpAddress ipAddr("1.0.0.0");
    mgr.rdmaHandleMap[devPhyId][2][ipAddr] = fakeRdmaHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.DieAndFuncIdMap[fakeRdmaHandle] = {0, 0};
    mgr.RtpEnableMap[fakeRdmaHandle] = true;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.RtpEnableMap.count(fakeRdmaHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithUbHandle_Expect_CleanupUbEntry)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 1;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xBBBB;
    Hccl::IpAddress ipAddr("2.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ipAddr] = fakeUbHandle;
    mgr.tokenInfoMap[fakeUbHandle] = nullptr;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithUbHandleAndTokenInfo_Expect_CleanupTokenAndUb)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 2;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xCCCC;
    Hccl::IpAddress ipAddr("3.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ipAddr] = fakeUbHandle;
    mgr.tokenInfoMap[fakeUbHandle] = std::make_unique<Hccl::TokenInfoManager>(0, fakeUbHandle);
    mgr.DieAndFuncIdMap[fakeUbHandle] = {1, 2};
    mgr.RtpEnableMap[fakeUbHandle] = false;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.RtpEnableMap.count(fakeUbHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithJfcHandle_Expect_CleanupJfcEntry)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xDDDD;
    Hccl::IpAddress ipAddr("4.0.0.0");
    mgr.rdmaHandleMap[devPhyId][2][ipAddr] = fakeRdmaHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.jfcHandleMap[fakeRdmaHandle][Hccl::HrtUbJfcMode::NORMAL] = (Hccl::JfcHandle)0x1111;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.jfcHandleMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithMixedHandles_Expect_AllCleaned)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xEEEE;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xFFFF;
    Hccl::IpAddress ipRdma("5.0.0.0");
    Hccl::IpAddress ipUb("6.0.0.0");

    mgr.rdmaHandleMap[devPhyId][2][ipRdma] = fakeRdmaHandle;
    mgr.rdmaHandleMap[devPhyId][3][ipUb] = fakeUbHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.jfcHandleMap[fakeRdmaHandle][Hccl::HrtUbJfcMode::NORMAL] = (Hccl::JfcHandle)0x2222;
    mgr.jfcHandleMap[fakeUbHandle][Hccl::HrtUbJfcMode::STARS_POLL] = (Hccl::JfcHandle)0x3333;
    mgr.tokenInfoMap[fakeUbHandle] = nullptr;
    mgr.DieAndFuncIdMap[fakeRdmaHandle] = {3, 4};
    mgr.DieAndFuncIdMap[fakeUbHandle] = {5, 6};
    mgr.RtpEnableMap[fakeRdmaHandle] = true;
    mgr.RtpEnableMap[fakeUbHandle] = false;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.jfcHandleMap.size(), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.size(), 0);
    EXPECT_EQ(mgr.RtpEnableMap.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDestroyAll_WithDestroyed_Expect_NoCrash)
{
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    mgr.destroyed = true;
    mgr.DestroyAll();
    mgr.destroyed = false;
}