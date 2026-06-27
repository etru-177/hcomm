/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"

#include "string_util.h"

namespace hcomm{
namespace CcuRep {

CcuRepBlock::CcuRepBlock(const std::string &label) : label(label)
{
    type = CcuRepType::BLOCK;
}

std::vector<std::shared_ptr<CcuRepBase>> &CcuRepBlock::GetReps()
{
    return repVec;
}

void CcuRepBlock::Append(std::shared_ptr<CcuRepBase> rep)
{
    repVec.push_back(rep);
}

const std::string &CcuRepBlock::GetLabel() const
{
    return label;
}

uint16_t CcuRepBlock::InstrCount()
{
    instrCount = 0;
    for (const auto &repInBlock : repVec) {
        instrCount += repInBlock->InstrCount();
    }
    return instrCount;
}

bool CcuRepBlock::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    for (const auto &repInBlock : GetReps()) {
        repInBlock->Translate(instr, instrId, dep);
    }

    return translated;
}

std::string CcuRepBlock::Describe()
{
    return Hccl::StringFormat("RepBlock");
}

std::shared_ptr<CcuRepBase> CcuRepBlock::GetRepByInstrId(uint16_t instrId)
{
    for (const auto& rep : GetReps()) {
        const uint16_t repCount = rep->InstrCount();
        if (repCount == 0) {
            continue;
        }
        const uint16_t startId = rep->StartInstrId();
        const uint16_t endId = startId + repCount - 1;
        if (instrId >= startId && instrId <= endId) {
            return rep;
        }
    }
    return nullptr;
}

}; // namespace CcuRep
}; // namespace hcomm