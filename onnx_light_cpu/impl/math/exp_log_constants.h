// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu::detail {

constexpr float kExpHi32 = 88.3762626647949f;
// Half of the smallest subnormal: exp below this boundary rounds to zero.
constexpr float kExpLo32 = -103.97208f;
constexpr float kSmallestNormal32 = 1.17549435e-38f;
constexpr float kSubnormalScale32 = 8388608.0f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpC1_32 = 0.693359375f;
constexpr float kExpC2_32 = -2.12194440e-4f;
constexpr float kExpP0_32 = 1.9875691500e-4f;
constexpr float kExpP1_32 = 1.3981999507e-3f;
constexpr float kExpP2_32 = 8.3334519073e-3f;
constexpr float kExpP3_32 = 4.1665795894e-2f;
constexpr float kExpP4_32 = 1.6666665459e-1f;
constexpr float kExpP5_32 = 5.0000001201e-1f;
constexpr float kSqrtHf = 0.707106781186547524f;
constexpr float kLogP0_32 = 7.0376836292e-2f;
constexpr float kLogP1_32 = -1.1514610310e-1f;
constexpr float kLogP2_32 = 1.1676998740e-1f;
constexpr float kLogP3_32 = -1.2420140846e-1f;
constexpr float kLogP4_32 = 1.4249322787e-1f;
constexpr float kLogP5_32 = -1.6668057665e-1f;
constexpr float kLogP6_32 = 2.0000714765e-1f;
constexpr float kLogP7_32 = -2.4999993993e-1f;
constexpr float kLogP8_32 = 3.3333331174e-1f;
constexpr float kLogQ1_32 = -2.12194440e-4f;
constexpr float kLogQ2_32 = 0.693359375f;

constexpr double kExpHi64 = 709.78271289338399673222;
constexpr double kExpLo64 = -708.39641853226410622441;
constexpr double kLog2e = 1.4426950408889634073599;
constexpr double kExpC1_64 = 6.93145751953125e-1;
constexpr double kExpC2_64 = 1.42860682030941723212e-6;
constexpr double kExpP0_64 = 1.26177193074810590878e-4;
constexpr double kExpP1_64 = 3.02994407707441961300e-2;
constexpr double kExpP2_64 = 9.99999999999999999910e-1;
constexpr double kExpQ0_64 = 3.00198505138664455042e-6;
constexpr double kExpQ1_64 = 2.52448340349684104192e-3;
constexpr double kExpQ2_64 = 2.27265548208155028766e-1;
constexpr double kExpQ3_64 = 2.00000000000000000005e0;
constexpr double kSqrtH64 = 0.70710678118654752440;
constexpr double kLogP0_64 = 1.01875663804580931796e-4;
constexpr double kLogP1_64 = 4.97494994976747001425e-1;
constexpr double kLogP2_64 = 4.70579119878881725854e0;
constexpr double kLogP3_64 = 1.44989225341610930846e1;
constexpr double kLogP4_64 = 1.79368678507819816313e1;
constexpr double kLogP5_64 = 7.70838733755885391666e0;
constexpr double kLogQ0_64 = 1.12873587189167450590e1;
constexpr double kLogQ1_64 = 4.52279145837532221105e1;
constexpr double kLogQ2_64 = 8.29875266912776603211e1;
constexpr double kLogQ3_64 = 7.11544750618563894466e1;
constexpr double kLogQ4_64 = 2.31251620126765340583e1;
constexpr double kLogC1_64 = 2.121944400546905827679e-4; // subtracted
constexpr double kLogC2_64 = 0.693359375;                // added

} // namespace onnx_light_cpu::detail
