#include "aicpu_raw_ub_channel.h"

#include <cstddef>

#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "endpoint.h"
#include "exception_handler.h"
#include "hcomm_c_adpt.h"
#include "local_ub_rma_buffer.h"
#include "mem_transport_common.h"
#include "orion_adpt_utils.h"
#include "orion_adapter_hccp.h"
#include "string_util.h"

namespace hcomm {
namespace {
#pragma pack(push, 8)
union RawUbEid {
    uint8_t raw[COMM_ADDR_EID_LEN];
    uint64_t align8[2];
};

struct RawUbJettyKey {
    RawUbEid eid;
    uint32_t uasid;
    uint32_t id;
    uint32_t transportMode;
};
#pragma pack(pop)

static_assert(sizeof(RawUbEid) == 16U, "raw URMA EID must be 16 bytes");
static_assert(sizeof(RawUbJettyKey) == 32U, "raw URMA Jetty key must match RsJettyKeyInfo");
static_assert(offsetof(RawUbJettyKey, transportMode) == 24U, "raw URMA transport mode offset mismatch");

std::vector<char> PackRemoteBuffer(const HcommRawUbPeerDesc &peer)
{
    // Public liburma exports the raw 28-bit token_id. Hcomm's AICPU SQE ABI
    // consumes the 20-bit segment object id, matching RaUbAllocTokenIdHandle.
    const u32 tokenId = peer.tokenId >> Hccl::URMA_TOKEN_ID_RIGHT_SHIFT;
    Hccl::BinaryStream stream;
    stream << peer.gva << peer.bytes << tokenId << peer.tokenValue << UINT32_MAX;
    std::vector<char> out;
    stream.Dump(out);
    return out;
}

struct LocalBuffers {
    uint32_t count = 0;
    std::vector<char> data;
};

LocalBuffers PackLocalBuffers(EndpointHandle endpoint)
{
    std::shared_ptr<Hccl::LocalUbRmaBuffer>* handles = nullptr;
    uint32_t count = 0;
    (void)HcommMemGetAllMemHandles(endpoint, reinterpret_cast<void**>(&handles), &count);

    Hccl::BinaryStream stream;
    for (uint32_t index = 0; index < count; ++index) {
        const auto& buffer = handles[index];
        stream << static_cast<uint64_t>(buffer->GetAddr()) << static_cast<uint64_t>(buffer->GetSize())
               << buffer->GetTokenId() << buffer->GetTokenValue();
    }
    LocalBuffers result;
    result.count = count;
    stream.Dump(result.data);
    return result;
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

    const auto localWireEid = localAddr_.GetEid();
    const auto remoteWireEid = remoteAddr_.GetEid();
    const auto localSqeEid = localAddr_.GetReverseEid();
    const auto remoteSqeEid = remoteAddr_.GetReverseEid();
    HCCL_INFO("[AicpuRawUbChannel] EID order localWire[%s] localSqe[%s] "
        "remoteWire[%s] remoteSqe[%s].",
        Hccl::Bytes2hex(localWireEid.raw, sizeof(localWireEid.raw)).c_str(),
        Hccl::Bytes2hex(localSqeEid.raw, sizeof(localSqeEid.raw)).c_str(),
        Hccl::Bytes2hex(remoteWireEid.raw, sizeof(remoteWireEid.raw)).c_str(),
        Hccl::Bytes2hex(remoteSqeEid.raw, sizeof(remoteSqeEid.raw)).c_str());

    connection_ = std::make_unique<Hccl::DevUbCtpConnection>(endpoint->GetRdmaHandle(), localAddr_, remoteAddr_,
        Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::STARS_POLL);
    return HCCL_SUCCESS;
}

ChannelStatus AicpuRawUbChannel::GetStatus()
{
    RawUbJettyKey key{};
    (void)memcpy_s(key.eid.raw, sizeof(key.eid.raw), peer_.eid, sizeof(peer_.eid));
    key.uasid = peer_.uasid;
    key.id = peer_.jettyId;
    key.transportMode = peer_.transportMode;
    const bool ready = connection_->ConnectRaw(reinterpret_cast<const u8 *>(&key), sizeof(key), peer_.tokenValue);
    HCCL_INFO("[AicpuRawUbChannel] peer jettyId[%u] uasid[%u] transport[%u] token[0x%x] "
        "rawTokenId[%u] hcommTokenId[%u] gva[0x%llx] bytes[%llu] segmentBytes[%u] ready[%d].",
        peer_.jettyId, peer_.uasid, peer_.transportMode, peer_.tokenValue, peer_.tokenId,
        peer_.tokenId >> Hccl::URMA_TOKEN_ID_RIGHT_SHIFT, peer_.gva, peer_.bytes, peer_.segmentBytes, ready);
    if (ready) {
        if (remoteMemHandle_ == 0) {
            const auto imported = Hccl::HrtRaUbRemoteMemImport(
                reinterpret_cast<Endpoint *>(endpointHandle_)->GetRdmaHandle(),
                peer_.segment, peer_.segmentBytes, peer_.tokenValue);
            remoteMemHandle_ = imported.handle;
            remoteSegmentVa_ = imported.targetSegVa;
            HCCL_INFO("[AicpuRawUbChannel] remote segment imported. handle[0x%llx] targetSegVa[0x%llx] "
                "rawTokenId[%u] hcommTokenId[%u] tokenValue[0x%x].",
                remoteMemHandle_, remoteSegmentVa_, peer_.tokenId,
                peer_.tokenId >> Hccl::URMA_TOKEN_ID_RIGHT_SHIFT, peer_.tokenValue);
        }
        HCCL_INFO("[AicpuRawUbChannel] local AICPU connection[%s].", connection_->Describe().c_str());
    }
    return ready ? ChannelStatus::READY : ChannelStatus::INIT;
}

HcclResult AicpuRawUbChannel::H2DResPack(std::vector<char> &buffer)
{
    Hccl::BinaryStream stream;
    const u32 type = static_cast<u32>(Hccl::TransportType::UB);
    const u32 zero = 0;
    const u32 one = 1;
    std::vector<char> empty;
    const LocalBuffers localBuffers = PackLocalBuffers(endpointHandle_);
    std::vector<char> remoteBuffer = PackRemoteBuffer(peer_);
    std::vector<char> connectionId = connection_->GetUniqueId();
    HCCL_INFO("[AicpuRawUbChannel] H2D localBuffers[%u] peerJetty[%u] rawTokenId[%u] hcommTokenId[%u] "
        "tokenValue[0x%x] remoteSegmentVa[0x%llx] localConnection[%s].",
        localBuffers.count, peer_.jettyId, peer_.tokenId,
        peer_.tokenId >> Hccl::URMA_TOKEN_ID_RIGHT_SHIFT, peer_.tokenValue, remoteSegmentVa_,
        connection_->Describe().c_str());
    stream << type << zero << localBuffers.count << one << one;
    stream << empty << empty << localBuffers.data << remoteBuffer << connectionId;
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
HcclResult AicpuRawUbChannel::Clean()
{
    if (remoteMemHandle_ != 0) {
        auto *endpoint = reinterpret_cast<Endpoint *>(endpointHandle_);
        Hccl::HrtRaUbRemoteMemUnimport(endpoint->GetRdmaHandle(), remoteMemHandle_);
        remoteMemHandle_ = 0;
        remoteSegmentVa_ = 0;
    }
    return HCCL_SUCCESS;
}
HcclResult AicpuRawUbChannel::Resume() { return HCCL_SUCCESS; }
HcclResult AicpuRawUbChannel::NotifyRecord(uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::NotifyWait(uint32_t, uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::WriteWithNotify(void *, const void *, uint64_t, uint32_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::Write(void *, const void *, uint64_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::Read(void *, const void *, uint64_t) { return HCCL_E_NOT_SUPPORT; }
HcclResult AicpuRawUbChannel::ChannelFence() { return HCCL_E_NOT_SUPPORT; }

} // namespace hcomm
