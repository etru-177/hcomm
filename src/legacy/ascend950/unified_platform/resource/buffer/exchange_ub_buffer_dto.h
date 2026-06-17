/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCLV2_EXCHANGE_UB_BUFFER_DTO_H
#define HCCLV2_EXCHANGE_UB_BUFFER_DTO_H
#include <map>
#include <string>
#include "serializable.h"
#include "orion_adapter_hccp.h"
#include "hccl_one_sided_data.h"
namespace Hccl {

class ExchangeUbBufferDto : public Serializable { // UB RMA Buffer / Notify /CntNotify 需要交换的DTO
public:
    ExchangeUbBufferDto()
    {
    }

    ExchangeUbBufferDto(u64 addr, u64 size, u32 tokenValue, u32 tokenId, u32 keySize)
        : addr(addr), size(size), tokenValue(tokenValue), tokenId(tokenId), keySize(keySize)
    {
    }

    ExchangeUbBufferDto(u64 addr, u64 size, HcclMemType memType, const char *memInfo, u32 tokenValue, u32 tokenId, u32 keySize)
        : addr(addr), size(size), memType(memType), memInfo(memInfo), tokenValue(tokenValue), tokenId(tokenId), keySize(keySize)
    {
    }

    void Serialize(Hccl::BinaryStream &stream) override
    {
        stream << addr << size << memType << memInfo << tokenValue << tokenId << keySize << key << segVa;
    }

    void Deserialize(Hccl::BinaryStream &stream) override
    {
        stream >> addr >> size >> memType >> memInfo >> tokenValue >> tokenId >> keySize >> key >> segVa;
    }

    std::string Describe() const override
    {
        return StringFormat(
            "ExchangeUbBufferDto[addr=0x%llx, size=0x%llx keySize=%u memInfo %s]", addr, size, keySize, memInfo.c_str());
    }

    u64 addr{0};
    u64 size{0};
    HcclMemType memType;
    std::string memInfo;
    u8  key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenValue{0};
    u32 tokenId{0};
    u32 keySize{0};
    u64 segVa{0};
};

} // namespace Hccl

#endif