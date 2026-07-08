#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "urma_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "rdma_handle_manager.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "endpoint.h"
#include "urma_mem.h"
#include "adapter_rts_common.h"
#include "server_socket_manager.h"
#include "hccp_hdc_manager.h"
#include "ccu_channel_ctx_pool.h"

#define private public
using namespace hcomm;

class UrmaEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UrmaEndpointTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UrmaEndpointTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in UrmaEndpointTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        endpointDesc.protocol = COMM_PROTOCOL_UBC_TP;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        endpointDesc.loc.device.devPhyId = 0;

        MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UrmaEndpointTest TearDown" << std::endl;
    }

    EndpointDesc endpointDesc{};
};

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(endpoint->ServerSocketListen(60001), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_LocTypeNotDevice_Expect_HCCL_SUCCESS)
{
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);

    EXPECT_EQ(endpoint->ServerSocketListen(60001), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_StartListenFailed_Expect_Error)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(endpoint->ServerSocketListen(60001), HCCL_E_INTERNAL);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    uint32_t portValue = 60001;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    EXPECT_EQ(port, 60001);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_LocTypeNotDevice_Expect_HCCL_SUCCESS)
{
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);

    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_AlreadyListening_Expect_ReturnCachedPort)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    uint32_t cachedPort = 50001;
    endpoint->dynamicPort_ = cachedPort;

    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    EXPECT_EQ(port, cachedPort);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenReturnsPortZero_Expect_HCCL_E_NETWORK)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    uint32_t portValue = 0;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_E_NETWORK);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenReturnsInvalidPort_Expect_HCCL_E_NETWORK)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    uint32_t portValue = HCCL_INVALID_PORT;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_E_NETWORK);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenFailed_Expect_Error)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->ServerSocketGetListenPort(&port), HCCL_E_INTERNAL);
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_NullCtxHandle_Expect_False)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ctxHandle_ = nullptr;
    EXPECT_EQ(endpoint->IsCtxHandleValid(), false);
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_ValidHandle_Expect_True)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    void *fakeHandle = (void *)0x1234;
    endpoint->ctxHandle_ = fakeHandle;
    auto &mgr = Hccl::RdmaHandleManager::GetInstance();
    mgr.destroyed = false;
    mgr.activeHandles_.insert(
        static_cast<Hccl::RdmaHandle>(fakeHandle));
    EXPECT_EQ(endpoint->IsCtxHandleValid(), true);
    mgr.activeHandles_.erase(
        static_cast<Hccl::RdmaHandle>(fakeHandle));
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_HandleDeInited_Expect_False)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    void *fakeHandle = (void *)0x5678;
    endpoint->ctxHandle_ = fakeHandle;
    EXPECT_EQ(endpoint->IsCtxHandleValid(), false);
}

TEST_F(UrmaEndpointTest, Ut_When_GetAsyncEvents_InvalidCtxHandle_Expect_InternalError)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ctxHandle_ = (void *)0x9999;
    uint32_t interfaceVersion = 2;
    MOCKER(RaGetInterfaceVersion).stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion, sizeof(interfaceVersion)))
        .will(returnValue(0));
    struct AsyncEvent events[1];
    uint32_t num = 1;
    EXPECT_EQ(endpoint->GetAsyncEvents(0, events, num), HCCL_E_INTERNAL);
}