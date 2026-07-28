#include "aicpu_raw_ub_channel.h"

#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "endpoint.h"
#include "exception_handler.h"
#include "mem_transport_common.h"
#include "orion_adpt_utils.h"

namespace hcomm {
namespace {
struct RawUbJettyKey {
    uint8_t eid[COMM_ADDR_EID_LEN];
    uint32_t uasid;
    uint32_t id;
    uint32_t transportMode;
};

std::vector<char> PackRemoteBuffer(const HcommRawUbPeerDesc &peer)
{
    Hccl::BinaryStream stream;
    stream << peer.gva << peer.bytes << peer.tokenId << peer.tokenValue << UINT32_MAX;
    std::vector<char> out;
    stream.Dump(out);
    return out;
}
} // namespace

AicpuRawUbChannel::AicpuRawUbChannel(EndpointHandle endpointHandle, const HcommRawUbPeerDesc &peer)
    : endpointHandle_(endpointHandle), peer_(peer) {}

HcclResult AicpuRawUbChannel::Init()
{
    auto *endpoint = reinterpret_cast<Endpoint *>(endpointHandle_);
    CHK_PTR_NULL(endpoint);
    const EndpointDesc local = endpoint->GetEndpointDesc();
    CHK_RET(CommAddrToIpAddress(local.commAddr, localAddr_));

    CommAddr remote{};
    remote.type = COMM_ADDR_TYPE_EID;
    CHK_SAFETY_FUNC_RET(memcpy_s(remote.eid, sizeof(remote.eid), peer_.eid, sizeof(peer_.eid)));
    CHK_RET(CommAddrToIpAddress(remote, remoteAddr_));

    connection_ = std::make_unique<Hccl::DevUbTpConnection>(endpoint->GetRdmaHandle(), localAddr_, remoteAddr_,
        Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::STARS_POLL);
    return HCCL_SUCCESS;
}

ChannelStatus AicpuRawUbChannel::GetStatus()
{
    RawUbJettyKey key{};
    (void)memcpy_s(key.eid, sizeof(key.eid), peer_.eid, sizeof(peer_.eid));
    key.uasid = peer_.uasid;
    key.id = peer_.jettyId;
    key.transportMode = peer_.transportMode;
    return connection_->ConnectRaw(reinterpret_cast<const u8 *>(&key), sizeof(key), peer_.tokenValue)
        ? ChannelStatus::READY : ChannelStatus::INIT;
}

HcclResult AicpuRawUbChannel::H2DResPack(std::vector<char> &buffer)
{
    Hccl::BinaryStream stream;
    const u32 type = static_cast<u32>(Hccl::TransportType::UB);
    const u32 zero = 0;
    const u32 one = 1;
    std::vector<char> empty;
    std::vector<char> remoteBuffer = PackRemoteBuffer(peer_);
    std::vector<char> connectionId = connection_->GetUniqueId();
    stream << type << zero << zero << one << one;
    stream << empty << empty << empty << remoteBuffer << connectionId;
    std::vector<char> uniqueId;
    stream.Dump(uniqueId);

    // AicpuChannelProcess first extracts a vector<char> transport unique id
    // from the STREAM module, matching AicpuTsUboeChannel::H2DResPack.
    Hccl::BinaryStream moduleStream;
    moduleStream << uniqueId;
    std::vector<char> moduleData;
    moduleStream.Dump(moduleData);

    std::vector<Hccl::ModuleData> modules(Hccl::AicpuResMgrType::__COUNT__);
    modules[Hccl::AicpuResMgrType::STREAM].data = moduleData;
    Hccl::AicpuResPackageHelper helper;
    buffer = helper.GetPackedData(modules);
    return HCCL_SUCCESS;
}

HcclResult AicpuRawUbChannel::GetNotifyNum(uint32_t *notifyNum) const { *notifyNum = 0; return HCCL_SUCCESS; }
HcclResult AicpuRawUbChannel::GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos)
{ *memNum = 0; *remoteMem = nullptr; *memInfos = nullptr; return HCCL_SUCCESS; }
HcclResult AicpuRawUbChannel::Clean() { return HCCL_SUCCESS; }
HcclResult AicpuRawUbChannel::Resume() { return HCCL_SUCCESS; }
HcclResult AicpuRawUbChannel::NotifyRecord(uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::NotifyWait(uint32_t, uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::WriteWithNotify(void *, const void *, uint64_t, uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::Write(void *, const void *, uint64_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::Read(void *, const void *, uint64_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::ChannelFence() { return HCCL_E_NOT_SUPPORT; }

} // namespace hcomm
