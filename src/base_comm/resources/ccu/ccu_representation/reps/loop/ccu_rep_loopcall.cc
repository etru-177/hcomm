/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

CcuRepLoopCall::CcuRepLoopCall(const std::string &label) : label(label)
{
    type = CcuRepType::LOOP_CALL;
}

const std::string &CcuRepLoopCall::GetLabel() const
{
    return label;
}

void CcuRepLoopCall::Reference(std::shared_ptr<CcuRepLoopBlock> refRep)
{
    loopBlock = refRep;
}

void CcuRepLoopCall::SetInArg(const Variable &var)
{
    inArgCount++;
    inArgInstrCount++;
    inArgs.push_back(CcuRepArg(var));
}

void CcuRepLoopCall::SetInArg(const std::vector<Variable> &varList)
{
    inArgCount += varList.size();
    inArgInstrCount += varList.size();
    inArgs.push_back(CcuRepArg(varList));
}

void CcuRepLoopCall::SetInArg(const Memory &mem)
{
    inArgCount++;
    inArgInstrCount += 2; // 传递Memory需要2条指令
    inArgs.push_back(CcuRepArg(mem));
}

void CcuRepLoopCall::SetInArg(const std::vector<Memory> &memList)
{
    inArgCount += memList.size();
    inArgInstrCount += memList.size() * 2; // 传递Memory需要2条指令
    inArgs.push_back(CcuRepArg(memList));
}

/*【新增】*/
void CcuRepLoopCall::SetInArg(const LocalAddr &addr)
{
    inArgCount++;
    inArgInstrCount += 2; // 传递LocalAddr需要2条指令
    inArgs.push_back(CcuRepArg(addr));
}

void CcuRepLoopCall::SetInArg(const std::vector<LocalAddr> &addrList)
{
    inArgCount += addrList.size();
    inArgInstrCount += addrList.size() * 2; // 传递LocalAddr需要2条指令
    inArgs.push_back(CcuRepArg(addrList));
}

void CcuRepLoopCall::SetInArg(const RemoteAddr &addr)
{
    inArgCount++;
    inArgInstrCount += 2; // 传递RemoteAddr需要2条指令
    inArgs.push_back(CcuRepArg(addr));
}

void CcuRepLoopCall::SetInArg(const std::vector<RemoteAddr> &addrList)
{
    inArgCount += addrList.size();
    inArgInstrCount += addrList.size() * 2; // 传递RemoteAddr需要2条指令
    inArgs.push_back(CcuRepArg(addrList));
}

uint16_t CcuRepLoopCall::InstrCount()
{
    instrCount = inArgInstrCount;
    return instrCount;
}

bool CcuRepLoopCall::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    this->instrId = instrId;
    translated    = true;

    Hccl::CHECK_NULLPTR(loopBlock, "[CcuRepLoopCall::Translate] LoopBlock is nullptr!");

    if (!loopBlock->Translated()) {
        Hccl::THROW<Hccl::CcuApiException>("Reference To Invalid LoopBlock");
    }

    for (uint32_t i = 0; i < inArgs.size(); i++) {
        if (inArgs[i].type == CcuArgType::VARIABLE && loopBlock->GetArg(i).type == CcuArgType::VARIABLE) {
            LoadXXInstr(instr++, loopBlock->GetArg(i).var.Id(), inArgs[i].var.Id(), dep.reserveXnId);
        } else if (inArgs[i].type == CcuArgType::VARIABLE_LIST
                   && loopBlock->GetArg(i).type == CcuArgType::VARIABLE_LIST) {
            if (inArgs[i].varList.size() != loopBlock->GetArg(i).varList.size()) {
                Hccl::THROW<Hccl::CcuApiException>("Mismatched Arg Size");
            }
            for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                LoadXXInstr(instr++, loopBlock->GetArg(i).varList[j].Id(), inArgs[i].varList[j].Id(), dep.reserveXnId);
            }
        } else if (inArgs[i].type == CcuArgType::MEMORY && loopBlock->GetArg(i).type == CcuArgType::MEMORY) {
            LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).mem.addr.Id(), inArgs[i].mem.addr.Id(), dep.reserveGsaId);
            LoadXXInstr(instr++, loopBlock->GetArg(i).mem.token.Id(), inArgs[i].mem.token.Id(), dep.reserveXnId);
        } else if (inArgs[i].type == CcuArgType::LOCAL_ADDR && loopBlock->GetArg(i).type == CcuArgType::LOCAL_ADDR) {
            LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).localAddr.addr.Id(), inArgs[i].localAddr.addr.Id(), dep.reserveGsaId);
            LoadXXInstr(instr++, loopBlock->GetArg(i).localAddr.token.Id(), inArgs[i].localAddr.token.Id(), dep.reserveXnId);
        } else if (inArgs[i].type == CcuArgType::REMOTE_ADDR && loopBlock->GetArg(i).type == CcuArgType::REMOTE_ADDR) {
            LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).remoteAddr.addr.Id(), inArgs[i].remoteAddr.addr.Id(), dep.reserveGsaId);
            LoadXXInstr(instr++, loopBlock->GetArg(i).remoteAddr.token.Id(), inArgs[i].remoteAddr.token.Id(), dep.reserveXnId);
        } else if (inArgs[i].type == CcuArgType::MEMORY_LIST && loopBlock->GetArg(i).type == CcuArgType::MEMORY_LIST) {
            if (inArgs[i].memList.size() != loopBlock->GetArg(i).memList.size()) {
                Hccl::THROW<Hccl::CcuApiException>("Mismatched Arg Size");
            }
            for (uint32_t j = 0; j < inArgs[i].memList.size(); j++) {
                LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).memList[j].addr.Id(), inArgs[i].memList[j].addr.Id(),
                                dep.reserveGsaId);
                LoadXXInstr(instr++, loopBlock->GetArg(i).memList[j].token.Id(), inArgs[i].memList[j].token.Id(),
                            dep.reserveXnId);
            }
        } else if (inArgs[i].type == CcuArgType::LOCAL_ADDR_LIST && loopBlock->GetArg(i).type == CcuArgType::LOCAL_ADDR_LIST) {
            if (inArgs[i].localAddrList.size() != loopBlock->GetArg(i).localAddrList.size()) {
                Hccl::THROW<Hccl::CcuApiException>("Mismatched Arg Size");
            }
            for (uint32_t j = 0; j < inArgs[i].localAddrList.size(); j++) {
                LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).localAddrList[j].addr.Id(), inArgs[i].localAddrList[j].addr.Id(),
                                dep.reserveGsaId);
                LoadXXInstr(instr++, loopBlock->GetArg(i).localAddrList[j].token.Id(), inArgs[i].localAddrList[j].token.Id(),
                            dep.reserveXnId);
            }
        } else if (inArgs[i].type == CcuArgType::REMOTE_ADDR_LIST && loopBlock->GetArg(i).type == CcuArgType::REMOTE_ADDR_LIST) {
            if (inArgs[i].remoteAddrList.size() != loopBlock->GetArg(i).remoteAddrList.size()) {
                Hccl::THROW<Hccl::CcuApiException>("Mismatched Arg Size");
            }
            for (uint32_t j = 0; j < inArgs[i].remoteAddrList.size(); j++) {
                LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).remoteAddrList[j].addr.Id(), inArgs[i].remoteAddrList[j].addr.Id(),
                                dep.reserveGsaId);
                LoadXXInstr(instr++, loopBlock->GetArg(i).remoteAddrList[j].token.Id(), inArgs[i].remoteAddrList[j].token.Id(),
                            dep.reserveXnId);
            }
        } else {
            Hccl::THROW<Hccl::CcuApiException>("Mismatched Arg Type");
        }
    }

    instrId += InstrCount();

    return translated;
}

std::string CcuRepLoopCall::Describe()
{
    return Hccl::StringFormat("LoopCall[%s]", label.c_str());
}

}; // namespace CcuRep
}; // namespace hcomm