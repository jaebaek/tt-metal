// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_ISINF_ISNAN_INCLUDE
#include "compute_kernel_api/eltwise_unary/isinf_isnan.h"
#endif

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#endif

#if SFPU_OP_EXP_INCLUDE
#include "compute_kernel_api/eltwise_unary/exp.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "compute_kernel_api/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/relu.h"
#endif

#if SFPU_OP_ELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/elu.h"
#endif

#if SFPU_OP_I0_INCLUDE
#include "compute_kernel_api/eltwise_unary/i0.h"
#endif

#if SFPU_OP_ERFINV_INCLUDE
#include "compute_kernel_api/eltwise_unary/erfinv.h"
#endif

#if SFPU_OP_NEG_INCLUDE
#include "compute_kernel_api/eltwise_unary/negative.h"
#endif

#if SFPU_OP_TRIG_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#endif

#if SFPU_OP_REVERSE_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/reverseops.h"
#endif

#if SFPU_OP_IDENTITY_INCLUDE
#include "compute_kernel_api/eltwise_unary/identity.h"
#endif

#if SFPU_OP_TYPECAST_INCLUDE
#include "compute_kernel_api/eltwise_unary/typecast.h"
#endif

#if SFPU_OP_RIGHT_SHIFT_INCLUDE
#include "compute_kernel_api/eltwise_unary/right_shift.h"
#endif

#if SFPU_OP_FLOOR_INCLUDE
#include "compute_kernel_api/eltwise_unary/floor.h"
#endif

#if SFPU_OP_LEFT_SHIFT_INCLUDE
#include "compute_kernel_api/eltwise_unary/left_shift.h"
#endif

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_SOFTPLUS_INCLUDE
#include "compute_kernel_api/eltwise_unary/softplus.h"
#endif

#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "compute_kernel_api.h"
#endif

#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "compute_kernel_api.h"
#endif
