/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include <climits>

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

CcuRepLoop::CcuRepLoop(const std::string &label, const Variable &loopParam) : label(label), loopParam(loopParam)
{
    type       = CcuRepType::LOOP;
    instrCount = 1; // loop翻译需要1条指令
}

const std::string &CcuRepLoop::GetLabel() const
{
    return label;
}

void CcuRepLoop::Reference(std::shared_ptr<CcuRepLoopBlock> refRep)
{
    loopBlock = refRep;
}

std::shared_ptr<CcuRepBase> CcuRepLoop::SetLoopParam(Executor executor, Variable var)
{
    return std::make_shared<CcuRepSetLoop>(loopParam, executor, var);
}

bool CcuRepLoop::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    Hccl::CHECK_NULLPTR(loopBlock, "[CcuRepLoop::Translate] LoopBlock is nullptr!");

    if (!loopBlock->Translated()) {
        Hccl::THROW<Hccl::CcuApiException>("Reference To Invalid LoopBlock");
    }

    uint16_t startInstrId = loopBlock->StartInstrId();
    uint16_t loopBlockInstrCount = loopBlock->InstrCount();
    if (loopBlockInstrCount == 0) {
        HCCL_ERROR("[CcuRepLoop][Translate] loopBlockInstrCount[%u] is 0, which causes underflow in endInstrId calculation.",
                    loopBlockInstrCount);
        return false;
    }
    if (startInstrId > USHRT_MAX - loopBlockInstrCount) {
        HCCL_ERROR("[CcuRepLoop][Translate] startInstrId[%u] + loopBlockInstrCount[%u] exceeds the maximum value of unsigned short int.",
                    startInstrId, loopBlockInstrCount);
        return false;
    }

    if (instrId > USHRT_MAX - instrCount) {
        HCCL_ERROR("[CcuRepLoop][Translate] instrId[%u] exceeds the maximum value of unsigned short int.", instrId);
        return false;
    }
    
    uint16_t endInstrId = startInstrId + loopBlockInstrCount - 1;

    LoopInstr(instr++, startInstrId, endInstrId, loopParam.Id());


    instrId += instrCount;

    return translated;
}

std::string CcuRepLoop::Describe()
{
    return Hccl::StringFormat("Loop reference to [%s]", label.c_str());
}

}; // namespace CcuRep
}; // namespace hcomm