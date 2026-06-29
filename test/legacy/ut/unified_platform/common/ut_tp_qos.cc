/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "tp_qos.h"
#include "hccp.h"

#include <cstring>
#include <string>

using namespace Hccl;

namespace {

const char *gStubHccnQosDscpCfg = nullptr;

int StubRaGetHccnCfgByCfgPtr(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr || gStubHccnQosDscpCfg == nullptr) {
        return -1;
    }
    const unsigned int len = static_cast<unsigned int>(std::strlen(gStubHccnQosDscpCfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, gStubHccnQosDscpCfg, len);
    *valueLen = len;
    return 0;
}

std::string BuildEightPairQosDscpCfg()
{
    std::string cfg;
    for (uint32_t qos = 0U; qos < 8U; ++qos) {
        if (qos > 0U) {
            cfg += ',';
        }
        cfg += std::to_string(qos);
        cfg += ':';
        cfg += std::to_string(10U + qos);
    }
    return cfg;
}

class TpQosTest : public testing::Test {
protected:
    void SetUp() override
    {
        gStubHccnQosDscpCfg = nullptr;
        MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgByCfgPtr));
    }

    void TearDown() override
    {
        gStubHccnQosDscpCfg = nullptr;
        GlobalMockObject::verify();
    }
};

} // namespace

TEST_F(TpQosTest, parse_dscp_production_cfg_qos0_expect_33)
{
    gStubHccnQosDscpCfg = "0:33,1:65";

    uint8_t dscp = 0U;
    EXPECT_TRUE(TpQosGetDscpByQosFromHccnCfg(0U, 0U, dscp));
    EXPECT_EQ(dscp, 33U);
}

TEST_F(TpQosTest, parse_dscp_production_cfg_qos1_expect_65)
{
    gStubHccnQosDscpCfg = "0:33,1:65";

    uint8_t dscp = 0U;
    EXPECT_TRUE(TpQosGetDscpByQosFromHccnCfg(0U, 1U, dscp));
    EXPECT_EQ(dscp, 65U);
}

TEST_F(TpQosTest, parse_dscp_production_cfg_missing_qos_expect_false)
{
    gStubHccnQosDscpCfg = "0:33,1:65";

    uint8_t dscp = 0U;
    EXPECT_FALSE(TpQosGetDscpByQosFromHccnCfg(0U, 2U, dscp));
}

TEST_F(TpQosTest, parse_dscp_eight_pairs_all_qos_expect_match)
{
    const std::string cfg = BuildEightPairQosDscpCfg();
    gStubHccnQosDscpCfg = cfg.c_str();

    for (uint32_t qos = 0U; qos < 8U; ++qos) {
        uint8_t dscp = 0U;
        ASSERT_TRUE(TpQosGetDscpByQosFromHccnCfg(0U, static_cast<uint8_t>(qos), dscp))
            << "qos=" << qos << " cfg=" << cfg;
        EXPECT_EQ(dscp, static_cast<uint8_t>(10U + qos)) << "qos=" << qos;
    }
}

TEST_F(TpQosTest, parse_dscp_nine_pairs_only_first_eight_supported)
{
    std::string cfg = BuildEightPairQosDscpCfg();
    cfg += ",8:18";
    gStubHccnQosDscpCfg = cfg.c_str();

    uint8_t dscp = 0U;
    EXPECT_FALSE(TpQosGetDscpByQosFromHccnCfg(0U, 8U, dscp));

    dscp = 0U;
    EXPECT_TRUE(TpQosGetDscpByQosFromHccnCfg(0U, 7U, dscp));
    EXPECT_EQ(dscp, 17U);
}

TEST_F(TpQosTest, parse_dscp_invalid_format_expect_false)
{
    gStubHccnQosDscpCfg = "0-33,1-65";

    uint8_t dscp = 0U;
    EXPECT_FALSE(TpQosGetDscpByQosFromHccnCfg(0U, 0U, dscp));
}
