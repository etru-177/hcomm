#ifndef AICPU_RAW_UB_CHANNEL_H
#define AICPU_RAW_UB_CHANNEL_H

#include "channel.h"
#include "dev_ub_connection.h"

namespace hcomm {

class AicpuRawUbChannel : public Channel {
public:
    AicpuRawUbChannel(EndpointHandle endpointHandle, const HcommRawUbPeerDesc &peer);
    HcclResult Init() override;
    HcclResult H2DResPack(std::vector<char> &buffer);
    HcclResult GetNotifyNum(uint32_t *notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t *memNum, CommMem **remoteMem, char ***memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult Clean() override;
    HcclResult Resume() override;
    HcommChannelKind GetChannelKind() const override { return HcommChannelKind::AICPU_RAW_UB; }
    HcclResult NotifyRecord(uint32_t) override;
    HcclResult NotifyWait(uint32_t, uint32_t) override;
    HcclResult WriteWithNotify(void *, const void *, uint64_t, uint32_t) override;
    HcclResult Write(void *, const void *, uint64_t) override;
    HcclResult Read(void *, const void *, uint64_t) override;
    HcclResult ChannelFence() override;
    HcclResult ExportLocalPeer(HcommRawUbLocalPeerDesc &desc) const;

private:
    EndpointHandle endpointHandle_;
    HcommRawUbPeerDesc peer_{};
    std::unique_ptr<Hccl::DevUbCtpConnection> connection_;
    Hccl::IpAddress localAddr_{};
    Hccl::IpAddress remoteAddr_{};
    Hccl::RemMemHandle remoteMemHandle_{0};
    uint64_t remoteSegmentVa_{0};
};

} // namespace hcomm
#endif
