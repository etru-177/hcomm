/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include "tsd.h"
#include "stdio.h"
#include "ut_dispatch.h"
#include "hccp_pub.h"
#include "securec.h"
#include "param.h"
#include "dl_hal_function.h"

extern int LltMain(int argc, char* argv);
extern int HccpAddToCgroup(int logicId);
extern int HccpParamParse(int argc, char *argv[], struct HccpInitParam *param);
extern int HccpSetLogInfo(struct HccpInitParam *param);
extern void RsApiDeinit(void);
extern int RsApiInit(void);
extern int HccpChangeNumOfFile();
int dlDrvGetDevNum(unsigned int *numDev);
int dlDrvDeviceGetPhyIdByIndex(unsigned int devIndex, unsigned int *phyId);

void TcNormal()
{

	return;
}

void TcAbnormal()
{

    return;
}

