#include "ggml.h"
#include "ggml-opencl.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// llamacppOpenCL11: only OpenCL 1.1 functions are called (CLBlast is no longer used)
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define CL_DMMV_LOCAL_SIZE 32

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 1
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

#define MULTILINE_QUOTE(...) #__VA_ARGS__
static std::string program_source = MULTILINE_QUOTE(

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

// The fp16 fields are declared as ushort (same size) and read with vload_half through a half
// pointer: without cl_khr_fp16, OpenCL C 1.1 only allows half as the pointee type of a pointer, and
// strict compilers reject half struct members (the NVIDIA compiler accepts them).
struct __attribute__ ((packed)) block_q4_0
{
    ushort d;
    uint8_t qs[QK4_0 / 2];
};

struct __attribute__ ((packed)) block_q4_1
{
    ushort d;
    ushort m;
    uint8_t qs[QK4_1 / 2];
};

struct __attribute__ ((packed)) block_q5_0
{
    ushort d;
    uint32_t qh;
    uint8_t qs[QK5_0 / 2];
};

struct __attribute__ ((packed)) block_q5_1
{
    ushort d;
    ushort m;
    uint32_t qh;
    uint8_t qs[QK5_1 / 2];
};

struct __attribute__ ((packed)) block_q8_0
{
    ushort d;
    int8_t qs[QK8_0];
};

struct __attribute__((packed)) block_q2_K
{
    uint8_t scales[16];
    uint8_t qs[64];
    ushort d;
    ushort dmin;
};

struct __attribute__((packed)) block_q3_K
{
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    ushort d;
};

struct __attribute__((packed)) block_q4_K
{
    ushort d;
    ushort dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct __attribute__((packed)) block_q5_K
{
    ushort d;
    ushort dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct __attribute__((packed)) block_q6_K
{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    ushort d;
};

__kernel void convert_fp16_to_fp32(__global half* x, __global float* y) {
    const uint i = get_global_id(0);

    y[i] = vload_half(0, &x[i]);
}

void dequantize_q4_0(__global const struct block_q4_0* x, const int ib, const int iqs, float* v0, float* v1) {
    const float d = vload_half(0, (__global const half *) &x[ib].d);

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    *v0 = (vi0 - 8)*d;
    *v1 = (vi1 - 8)*d;
}
void dequantize_q4_1(__global const struct block_q4_1* x, const int ib, const int iqs, float* v0, float* v1) {
    const float d = vload_half(0, (__global const half *) &x[ib].d);
    const float m = vload_half(0, (__global const half *) &x[ib].m);

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    *v0 = vi0*d + m;
    *v1 = vi1*d + m;
}
void dequantize_q5_0(__global const struct block_q5_0* x, const int ib, const int iqs, float* v0, float* v1) {
    const float d = vload_half(0, (__global const half *) &x[ib].d);

    uint32_t qh = x[ib].qh;

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0) - 16;
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1) - 16;

    *v0 = x0*d;
    *v1 = x1*d;
}
void dequantize_q5_1(__global const struct block_q5_1* x, const int ib, const int iqs, float* v0, float* v1) {
    const float d = vload_half(0, (__global const half *) &x[ib].d);
    const float m = vload_half(0, (__global const half *) &x[ib].m);

    uint32_t qh = x[ib].qh;

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0);
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1);

    *v0 = x0*d + m;
    *v1 = x1*d + m;
}
void dequantize_q8_0(__global const struct block_q8_0* x, const int ib, const int iqs, float* v0, float* v1) {
    const float d = vload_half(0, (__global const half *) &x[ib].d);

    const int8_t vi0 = x[ib].qs[iqs + 0];
    const int8_t vi1 = x[ib].qs[iqs + 1];

    *v0 = vi0*d;
    *v1 = vi1*d;
}
void convert_f16(__global half* x, const int ib, const int iqs, float* v0, float* v1){
    *v0 = vload_half(0, &x[ib + 0]);
    *v1 = vload_half(0, &x[ib + 1]);
}
);

static std::string k_quants_source = MULTILINE_QUOTE(
inline void get_scale_min_k4(int j, const __global uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4)
    {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__kernel void dequantize_block_q2_K(__global const struct block_q2_K *x, __global float *yy)
{
    const int i = get_group_id(0) + get_global_offset(0);
    const int tid = get_local_id(0);
    const int n = tid / 32;
    const int l = tid - 32 * n;
    const int is = 8 * n + l / 16;

    const uint8_t q = x[i].qs[32 * n + l];
    __global float *y = yy + get_group_id(0) * QK_K + 128 * n;

    const float dall = vload_half(0, (__global const half *) &x[i].d);
    const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

    y[l + 0] = dall * (x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is + 0] >> 4);
    y[l + 32] = dall * (x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is + 2] >> 4);
    y[l + 64] = dall * (x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is + 4] >> 4);
    y[l + 96] = dall * (x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is + 6] >> 4);
}

__kernel void dequantize_block_q3_K(__global const struct block_q3_K *x, __global float *yy)
{
    int r = get_local_id(0) / 4;
    int i = get_group_id(0) + get_global_offset(0);
    int tid = r / 2;
    int is0 = r % 2;
    int l0 = 16 * is0 + 4 * (get_local_id(0) % 4);
    int n = tid / 4;
    int j = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int is = 8 * n + 2 * j + is0;
    int shift = 2 * j;

    int8_t us = is < 4 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 8] >> 0) & 3) << 4)
              : is < 8 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 4] >> 2) & 3) << 4)
              : is < 12  ? (x[i].scales[is - 8] >> 4) | (((x[i].scales[is + 0] >> 4) & 3) << 4)
              : (x[i].scales[is - 8] >> 4) | (((x[i].scales[is - 4] >> 6) & 3) << 4);
    float d_all = vload_half(0, (__global const half *) &x[i].d);
    float dl = d_all * (us - 32);

    __global float *y = yy + get_group_id(0) * QK_K + 128 * n + 32 * j;
    const __global uint8_t *q = x[i].qs + 32 * n;
    const __global uint8_t *hm = x[i].hmask;

    for (int l = l0; l < l0 + 4; ++l)
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

__kernel void dequantize_block_q4_K(__global const struct block_q4_K *x, __global float *yy)
{
    const int i = get_group_id(0) + get_global_offset(0);
    const int tid = get_local_id(0);
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const int n = 4;

    __global float *y = yy + get_group_id(0) * QK_K + 64 * il + n * ir;

    const float dall = vload_half(0, (__global const half *) &x[i].d);
    const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

    __global const uint8_t *q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
    float d1 = dall * sc;
    float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
    float d2 = dall * sc;
    float m2 = dmin * m;
    for (int l = 0; l < n; ++l)
    {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
}

__kernel void dequantize_block_q5_K(__global const struct block_q5_K *x, __global float *yy)
{
    const int i = get_group_id(0) + get_global_offset(0);
    const int tid = get_local_id(0);
    const int il = tid / 16;
    const int ir = tid % 16;
    const int is = 2 * il;

    __global float *y = yy + get_group_id(0) * QK_K + 64 * il + 2 * ir;

    const float dall = vload_half(0, (__global const half *) &x[i].d);
    const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

    __global const uint8_t *ql = x[i].qs + 32 * il + 2 * ir;
    __global const uint8_t *qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[0] = d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1;
    y[1] = d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2;
}

__kernel void dequantize_block_q6_K(__global const struct block_q6_K *x, __global float *yy)
{
    const int i = get_group_id(0) + get_global_offset(0);
    const int tid = get_local_id(0);
    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;

    __global float *y = yy + get_group_id(0) * QK_K + 128 * ip + il;

    const float d = vload_half(0, (__global const half *) &x[i].d);

    __global const uint8_t *ql = x[i].ql + 64 * ip + il;
    const uint8_t qh = x[i].qh[32 * ip + il];
    __global const int8_t *sc = x[i].scales + is;

    y[0] = d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

__kernel void dequantize_mul_mat_vec_q2_K(__global const struct block_q2_K * xx, __local float* tmp, __global float* yy, __global float* dst, const int ncols) {

    const int row = get_group_id(0);

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row + get_global_offset(0);

    __global const struct block_q2_K * x = xx + ib0;

    const int tid = get_local_id(0)/K_QUANTS_PER_ITERATION;  // 0...31 or 0...15
    const int ix  = get_local_id(0)%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    tmp[16 * ix + tid] = 0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        __global const float   * y = yy + i * QK_K + y_offset;
        __global const uint8_t * q = x[i].qs + q_offset;

        const float dall = vload_half(0, (__global const half *) &x[i].d);
        const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

        __global const uint32_t * a = (__global const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp[16 * ix + tid] += dall * sum1 - dmin * sum2;

    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=16; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}

__kernel void dequantize_mul_mat_vec_q3_K(__global const struct block_q3_K * xx, __local float* tmp, __global float* yy, __global float* dst, const int ncols) {
    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int row = get_group_id(0);

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row + get_global_offset(0);

    __global const struct block_q3_K * x = xx + ib0;

    const int tid = get_local_id(0)/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = get_local_id(0)%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    tmp[16 * ix + tid] = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        __global const float   * y  = yy + i * QK_K + y_offset;
        __global const uint8_t * q = x[i].qs + q_offset;
        __global const uint8_t * h = x[i].hmask + l0;

        __global const uint16_t * a = (__global const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = vload_half(0, (__global const half *) &x[i].d);

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp[16 * ix + tid] += d * sum;

    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=16; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}

__kernel void dequantize_mul_mat_vec_q4_K(__global const struct block_q4_K * xx, __local float* tmp, __global float* yy, __global float* dst, const int ncols) {

    //to rename it later, just to test now
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int row = get_group_id(0);
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row + get_global_offset(0);

    const int tid = get_local_id(0)/K_QUANTS_PER_ITERATION;  // 0...15
    const int ix  = get_local_id(0)%K_QUANTS_PER_ITERATION;

    const int step = 8/K_QUANTS_PER_ITERATION;

    const int il  = tid/step;     // 0...3
    const int ir  = tid - step*il;// 0...3
    const int n   = 2*K_QUANTS_PER_ITERATION;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    __global const struct block_q4_K * x = xx + ib0;

    tmp[16 * ix + tid] = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        __global const uint8_t * q1 = x[i].qs + q_offset;
        __global const uint8_t * q2 = q1 + 64;
        __global const float   * y1 = yy + i*QK_K + y_offset;
        __global const float   * y2 = y1 + 128;

        const float dall = vload_half(0, (__global const half *) &x[i].d);
        const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

        __global const uint16_t * a = (__global const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 s = (float4)(0.f);
        float smin = 0;
        for (int l = 0; l < n; ++l) {
            s.x += y1[l] * (q1[l] & 0xF); s.y += y1[l+32] * (q1[l] >> 4);
            s.z += y2[l] * (q2[l] & 0xF); s.w += y2[l+32] * (q2[l] >> 4);
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp[16 * ix + tid] += dall * (s.x * sc[0] + s.y * sc[1] + s.z * sc[4] + s.w * sc[5]) - dmin * smin;

    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=16; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}

__kernel void dequantize_mul_mat_vec_q5_K(__global const struct block_q5_K * xx, __local float* tmp, __global float* yy, __global float* dst, const int ncols) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int row = get_group_id(0);
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row + get_global_offset(0);

    const int tid = get_local_id(0)/2;  // 0...15
    const int ix  = get_local_id(0)%2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    __global const struct block_q5_K * x = xx + ib0;

    tmp[16 * ix + tid] = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        __global const uint8_t * ql1 = x[i].qs + q_offset;
        __global const uint8_t * ql2 = ql1 + 64;
        __global const uint8_t * qh  = x[i].qh + l0;
        __global const float   * y1  = yy + i*QK_K + y_offset;
        __global const float   * y2  = y1 + 128;

        const float dall = vload_half(0, (__global const half *) &x[i].d);
        const float dmin = vload_half(0, (__global const half *) &x[i].dmin);

        __global const uint16_t * a = (__global const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 sum = (float4)(0.f);
        float smin = 0;
        for (int l = 0; l < n; ++l) {
            sum.x += y1[l+ 0] * ((ql1[l+ 0] & 0xF) + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + y1[l+16] * ((ql1[l+16] & 0xF) + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += y1[l+32] * ((ql1[l+ 0] >>  4) + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + y1[l+48] * ((ql1[l+16] >>  4) + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += y2[l+ 0] * ((ql2[l+ 0] & 0xF) + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + y2[l+16] * ((ql2[l+16] & 0xF) + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += y2[l+32] * ((ql2[l+ 0] >>  4) + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + y2[l+48] * ((ql2[l+16] >>  4) + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp[16 * ix + tid] += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;

    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=16; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}

__kernel void dequantize_mul_mat_vec_q6_K(__global const struct block_q6_K * xx, __local float* tmp, __global const float * yy, __global float * dst, const int ncols) {

    const int row = get_group_id(0);

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row + get_global_offset(0);

    __global const struct block_q6_K * x = xx + ib0;

    const int tid = get_local_id(0)/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = get_local_id(0)%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

\n#if K_QUANTS_PER_ITERATION == 1\n
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;

\n#else\n

    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;

\n#endif\n

    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    tmp[16 * ix + tid] = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        __global const float   * y  = yy + i * QK_K + y_offset;
        __global const uint8_t * ql = x[i].ql + ql_offset;
        __global const uint8_t * qh = x[i].qh + qh_offset;
        __global const int8_t  * s  = x[i].scales + s_offset;

        const float d = vload_half(0, (__global const half *) &x[i].d);

\n#if K_QUANTS_PER_ITERATION == 1\n
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp[16 * ix + tid] += sum;
\n#else\n
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp[16 * ix + tid] += sum;
\n#endif\n

    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=16; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}
);


std::string dequant_template = MULTILINE_QUOTE(
__kernel void KERNEL_NAME(__global X_TYPE* x, __global float* y) {
    const int i = get_group_id(0)*get_local_size(0) + get_local_id(0)*2;

    if (i >= get_global_size(0)) {
        return;
    }

    const uint qk = QUANT_K;
    const uint qr = QUANT_R;

    const int ib = i/qk + get_global_offset(0); // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    float v0, v1;
    DEQUANT_FUNC(x, ib, iqs, &v0, &v1);
    y[iybs + iqs + 0] = v0;
    y[iybs + iqs + y_offset] = v1;
}
);

std::string dequant_mul_mat_vec_template = MULTILINE_QUOTE(
__kernel void KERNEL_NAME(__global X_TYPE* x, __local float* tmp, __global float* y, __global float* dst, const int ncols) {
    const int local_size = get_local_size(0);
    const int row = get_group_id(0);
    const int tid = get_local_id(0);

    const uint qk = QUANT_K;
    const uint qr = QUANT_R;

    const int col_step = local_size * 2;
    const int y_offset = qr == 1 ? 1 : qk/2;

    x += get_global_offset(0);

    tmp[tid] = 0;

    for (int col = tid*2; col < ncols; col += col_step) {
        const int ib = (row*ncols + col)/qk; // block index
        const int iqs = (col%qk)/qr; // quant index
        const int iybs = col - col%qk; // y block start index

        // dequantize
        float v0, v1;
        DEQUANT_FUNC(x, ib, iqs, &v0, &v1);

        // matrix multiplication
        tmp[tid] += v0 * y[iybs + iqs + 0];
        tmp[tid] += v1 * y[iybs + iqs + y_offset];
    }

    // sum up partial sums and write back result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=local_size/2; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        dst[row] = tmp[0];
    }
}

);


std::string mul_template = MULTILINE_QUOTE(
__kernel void KERNEL_NAME(__global TYPE* x, const int x_offset, __global TYPE* y, const int y_offset, __global TYPE* dst, const int dst_offset, const int ky) {
    const int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    if (i >= get_global_size(0)) {
        return;
    }

    dst[dst_offset + i] = x[x_offset + i] * y[y_offset + i%ky];
}
);

std::string add_template = MULTILINE_QUOTE(
__kernel void add_f32(__global float * x, const int x_offset, __global float * y, const int y_offset, __global float * dst, const int dst_offset, const int ky) {
    const int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    if (i >= get_global_size(0)) {
        return;
    }

    dst[dst_offset + i] = x[x_offset + i] + y[y_offset + i%ky];
}
);

// Fast dequantize_mul_mat_vec (matrix x vector, used for token generation) kernels for Q4_K, Q5_K
// and Q6_K, tuned on a Fermi GPU (GeForce GT 720M, OpenCL 1.1) with examples/bench-dmmv.
// Layout: 8 lanes per row and 256-weight super-block, so every lane handles 32 weights of each
// super-block with 4-byte quant reads; (local_size / 8) rows per work-group. The scales and mins of
// Q4_K/Q5_K are decoded for the whole super-block at once with 32-bit masks. Q6_K super-blocks
// (210 bytes) are only 2-byte aligned, so each unaligned 32-bit value is built from two aligned
// reads. The super-block offset of the matrix slice is passed explicitly in `boff`.
std::string dmmv_kq_l8_source = MULTILINE_QUOTE(

inline float kq_sum4(const float4 v) {
    return (v.x + v.y) + (v.z + v.w);
}

// reduce the 8 partial sums of every row of the work-group and store one result per row
inline void kq_reduce8_store(__local float * tmp, const int lid, const int lane, const float acc,
                             const int row, const int nrows, __global float * dst) {
    tmp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 4; s > 0; s >>= 1) {
        if (lane < s) {
            tmp[lid] += tmp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < nrows) {
        dst[row] = tmp[lid];
    }
}

// 12 packed scale bytes of a Q4_K/Q5_K super-block -> the 8 scales and 8 mins (get_scale_min_k4)
inline void kq_scales_mins(__global const uchar * blk, float4 * sc_a, float4 * m_a, float4 * sc_b, float4 * m_b) {
    __global const uint * s = (__global const uint *) (blk + 4);
    const uint s0 = s[0];
    const uint s1 = s[1];
    const uint s2 = s[2];
    *sc_a = convert_float4(as_uchar4(s0 & 0x3F3F3F3Fu));
    *m_a  = convert_float4(as_uchar4(s1 & 0x3F3F3F3Fu));
    *sc_b = convert_float4(as_uchar4(( s2       & 0x0F0F0F0Fu) | ((s0 >> 2) & 0x30303030u)));
    *m_b  = convert_float4(as_uchar4(((s2 >> 4) & 0x0F0F0F0Fu) | ((s1 >> 2) & 0x30303030u)));
}

// one 64-weight part of a Q4_K/Q5_K super-block: 4 low-nibble weights (sub-block 2k) at yp and
// 4 high-nibble weights (sub-block 2k + 1) at yp + 32
inline void kq_part(const float4 qlo, const float4 qhi, __global const float * yp,
                    const float sclo, const float schi, const float mlo, const float mhi,
                    float * sd, float * sm) {
    const float4 ylo = vload4(0, yp);
    const float4 yhi = vload4(0, yp + 32);
    *sd += sclo * dot(qlo, ylo) + schi * dot(qhi, yhi);
    *sm += mlo  * kq_sum4(ylo)  + mhi  * kq_sum4(yhi);
}

// block_q4_K (144 bytes): half d, half dmin, uchar scales[12], uchar qs[128]
__kernel void dequantize_mul_mat_vec_q4_K_l8(__global const uchar * xq_base, const ulong xoff,
                                             __global const char * y_base, const ulong yoff,
                                             __global char * dst_base, const ulong doff,
                                             const int ncols, const int nrows, __local float * tmp) {
    __global const uchar * xq  = xq_base + xoff;
    __global const float * y   = (__global const float *) (y_base + yoff);
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row  = get_group_id(0) * (get_local_size(0) >> 3) + (lid >> 3);
    const int nb   = ncols / QK_K;

    float acc = 0.0f;
    if (row < nrows) {
        __global const uchar * blk = xq + (size_t) row * nb * 144;
        __global const float * yb  = y + 4*j;
        for (int i = 0; i < nb; ++i, blk += 144, yb += QK_K) {
            const float2 dm = vload_half2(0, (__global const half *) blk);
            float4 sc_a, m_a, sc_b, m_b;
            kq_scales_mins(blk, &sc_a, &m_a, &sc_b, &m_b);
            // part k (64 weights) uses the quant word 8k + j
            __global const uint * qs = (__global const uint *) (blk + 16);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
            float sd = 0.0f;
            float sm = 0.0f;
            kq_part(convert_float4(as_uchar4(w0 & 0x0F0F0F0Fu)), convert_float4(as_uchar4((w0 >> 4) & 0x0F0F0F0Fu)), yb +   0, sc_a.x, sc_a.y, m_a.x, m_a.y, &sd, &sm);
            kq_part(convert_float4(as_uchar4(w1 & 0x0F0F0F0Fu)), convert_float4(as_uchar4((w1 >> 4) & 0x0F0F0F0Fu)), yb +  64, sc_a.z, sc_a.w, m_a.z, m_a.w, &sd, &sm);
            kq_part(convert_float4(as_uchar4(w2 & 0x0F0F0F0Fu)), convert_float4(as_uchar4((w2 >> 4) & 0x0F0F0F0Fu)), yb + 128, sc_b.x, sc_b.y, m_b.x, m_b.y, &sd, &sm);
            kq_part(convert_float4(as_uchar4(w3 & 0x0F0F0F0Fu)), convert_float4(as_uchar4((w3 >> 4) & 0x0F0F0F0Fu)), yb + 192, sc_b.z, sc_b.w, m_b.z, m_b.w, &sd, &sm);
            acc += dm.x * sd - dm.y * sm;
        }
    }
    kq_reduce8_store(tmp, lid, j, acc, row, nrows, dst);
}

// block_q5_K (176 bytes): half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128]
__kernel void dequantize_mul_mat_vec_q5_K_l8(__global const uchar * xq_base, const ulong xoff,
                                             __global const char * y_base, const ulong yoff,
                                             __global char * dst_base, const ulong doff,
                                             const int ncols, const int nrows, __local float * tmp) {
    __global const uchar * xq  = xq_base + xoff;
    __global const float * y   = (__global const float *) (y_base + yoff);
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row  = get_group_id(0) * (get_local_size(0) >> 3) + (lid >> 3);
    const int nb   = ncols / QK_K;

    float acc = 0.0f;
    if (row < nrows) {
        __global const uchar * blk = xq + (size_t) row * nb * 176;
        __global const float * yb  = y + 4*j;
        for (int i = 0; i < nb; ++i, blk += 176, yb += QK_K) {
            const float2 dm = vload_half2(0, (__global const half *) blk);
            float4 sc_a, m_a, sc_b, m_b;
            kq_scales_mins(blk, &sc_a, &m_a, &sc_b, &m_b);
            // qh byte l (l = 4j..4j+3) holds the 5th bit of weight l of every sub-block (bit k -> sub-block k)
            const uint hq = ((__global const uint *) (blk + 16))[j];
            __global const uint * qs = (__global const uint *) (blk + 48);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
            float sd = 0.0f;
            float sm = 0.0f;
            kq_part(convert_float4(as_uchar4(( w0       & 0x0F0F0F0Fu) | (((hq >> 0) & 0x01010101u) << 4))),
                    convert_float4(as_uchar4(((w0 >> 4) & 0x0F0F0F0Fu) | (((hq >> 1) & 0x01010101u) << 4))),
                    yb +   0, sc_a.x, sc_a.y, m_a.x, m_a.y, &sd, &sm);
            kq_part(convert_float4(as_uchar4(( w1       & 0x0F0F0F0Fu) | (((hq >> 2) & 0x01010101u) << 4))),
                    convert_float4(as_uchar4(((w1 >> 4) & 0x0F0F0F0Fu) | (((hq >> 3) & 0x01010101u) << 4))),
                    yb +  64, sc_a.z, sc_a.w, m_a.z, m_a.w, &sd, &sm);
            kq_part(convert_float4(as_uchar4(( w2       & 0x0F0F0F0Fu) | (((hq >> 4) & 0x01010101u) << 4))),
                    convert_float4(as_uchar4(((w2 >> 4) & 0x0F0F0F0Fu) | (((hq >> 5) & 0x01010101u) << 4))),
                    yb + 128, sc_b.x, sc_b.y, m_b.x, m_b.y, &sd, &sm);
            kq_part(convert_float4(as_uchar4(( w3       & 0x0F0F0F0Fu) | (((hq >> 6) & 0x01010101u) << 4))),
                    convert_float4(as_uchar4(((w3 >> 4) & 0x0F0F0F0Fu) | (((hq >> 7) & 0x01010101u) << 4))),
                    yb + 192, sc_b.z, sc_b.w, m_b.z, m_b.w, &sd, &sm);
            acc += dm.x * sd - dm.y * sm;
        }
    }
    kq_reduce8_store(tmp, lid, j, acc, row, nrows, dst);
}

// unaligned 32-bit value of a 2-byte aligned Q6_K super-block from two aligned 32-bit reads
inline uint q6_ld_u32(__global const uint * pa, const int idx, const uint sh, const uint hmask) {
    return (pa[idx] >> sh) | ((pa[idx + 1] << 16) & hmask);
}

// one (half, selector) group of a Q6_K super-block: 4 low-nibble weights (y at yp) and 4 high-nibble
// weights (y at yp + 64), with y in global (q6_group_g) or local (q6_group_l) memory
inline float q6_group_g(const uint w, const uint hq, const int sel, __global const float * yp,
                        const float slo, const float shi) {
    const float4 ylo = vload4(0, yp);
    const float4 yhi = vload4(0, yp + 64);
    const float4 qlo = convert_float4(as_uchar4(( w       & 0x0F0F0F0Fu) | (((hq >> (2*sel))     & 0x03030303u) << 4)));
    const float4 qhi = convert_float4(as_uchar4(((w >> 4) & 0x0F0F0F0Fu) | (((hq >> (4 + 2*sel)) & 0x03030303u) << 4)));
    return slo * (dot(qlo, ylo) - 32.0f*kq_sum4(ylo)) + shi * (dot(qhi, yhi) - 32.0f*kq_sum4(yhi));
}

inline float q6_group_l(const uint w, const uint hq, const int sel, __local const float * yp,
                        const float slo, const float shi) {
    const float4 ylo = vload4(0, yp);
    const float4 yhi = vload4(0, yp + 64);
    const float4 qlo = convert_float4(as_uchar4(( w       & 0x0F0F0F0Fu) | (((hq >> (2*sel))     & 0x03030303u) << 4)));
    const float4 qhi = convert_float4(as_uchar4(((w >> 4) & 0x0F0F0F0Fu) | (((hq >> (4 + 2*sel)) & 0x03030303u) << 4)));
    return slo * (dot(qlo, ylo) - 32.0f*kq_sum4(ylo)) + shi * (dot(qhi, yhi) - 32.0f*kq_sum4(yhi));
}

// block_q6_K (210 bytes): uchar ql[128], uchar qh[64], char scales[16], half d.
// Lane j handles l = 4j..4j+3 of both 128-weight halves and both selectors (ql[l] / ql[l + 32]).
// 210 = 2 (mod 4) and the buffer start is 4-byte aligned, so super-block n starts at an address
// = 2*(n & 1) (mod 4): the alignment is computed with 32-bit integers and the block pointer is just
// incremented. (Computing a 64-bit byte offset per block took 3 more registers, 43 instead of 40,
// which on Fermi leaves room for 2 instead of 3 work-groups of 256 and cost 20-27% speed.)
__kernel void dequantize_mul_mat_vec_q6_K_l8(__global const uchar * xq_base, const ulong xoff,
                                             __global const char * y_base, const ulong yoff,
                                             __global char * dst_base, const ulong doff,
                                             const int ncols, const int nrows, __local float * tmp) {
    __global const uchar * xq  = xq_base + xoff; // xoff must be a multiple of 4
    __global const float * y   = (__global const float *) (y_base + yoff);
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row  = get_group_id(0) * (get_local_size(0) >> 3) + (lid >> 3);
    const int nb   = ncols / QK_K;
    const int l    = 4*j;
    const int is   = j >> 2;

    float acc = 0.0f;
    if (row < nrows) {
        const int n0 = row * nb; // index of the first super-block of the row (relative to the 4-byte aligned xq)
        __global const uchar * blk = xq + (size_t) n0 * 210;
        for (int i = 0; i < nb; ++i, blk += 210) {
            const uint   odd   = (uint) ((n0 + i) & 1) << 1; // 0 or 2
            const uint   sh    = odd << 3;                   // 0 or 16
            const uint   hmask = 0u - (odd >> 1);            // 0 or 0xFFFFFFFF
            __global const uint  * pa  = (__global const uint *) (blk - odd);
            __global const char  * sc  = (__global const char *) (blk + 192);
            __global const float * yb  = y + i*QK_K + l;
            const float d  = vload_half(0, (__global const half *) (blk + 208));
            const uint  h0 = q6_ld_u32(pa, (128 + l)/4, sh, hmask);
            const uint  h1 = q6_ld_u32(pa, (160 + l)/4, sh, hmask);
            float sum = 0.0f;
            sum += q6_group_g(q6_ld_u32(pa, ( 0 + l)/4, sh, hmask), h0, 0, yb +   0, (float) sc[ 0 + is], (float) sc[ 4 + is]);
            sum += q6_group_g(q6_ld_u32(pa, (32 + l)/4, sh, hmask), h0, 1, yb +  32, (float) sc[ 2 + is], (float) sc[ 6 + is]);
            sum += q6_group_g(q6_ld_u32(pa, (64 + l)/4, sh, hmask), h1, 0, yb + 128, (float) sc[ 8 + is], (float) sc[12 + is]);
            sum += q6_group_g(q6_ld_u32(pa, (96 + l)/4, sh, hmask), h1, 1, yb + 160, (float) sc[10 + is], (float) sc[14 + is]);
            acc += d * sum;
        }
    }
    kq_reduce8_store(tmp, lid, j, acc, row, nrows, dst);
}

// same as dequantize_mul_mat_vec_q6_K_l8, but y is first staged in local memory (yl, ncols floats);
// faster for short rows (ncols <= 2048), e.g. the output matrix of small models
__kernel void dequantize_mul_mat_vec_q6_K_l8_ly(__global const uchar * xq_base, const ulong xoff,
                                                __global const char * y_base, const ulong yoff,
                                                __global char * dst_base, const ulong doff,
                                                const int ncols, const int nrows, __local float * tmp,
                                                __local float * yl) {
    __global const uchar * xq  = xq_base + xoff; // xoff must be a multiple of 4
    __global const float * y   = (__global const float *) (y_base + yoff);
    __global       float * dst = (__global float *) (dst_base + doff);
    for (int c = get_local_id(0); c < ncols/4; c += get_local_size(0)) {
        vstore4(vload4(c, y), c, yl);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row  = get_group_id(0) * (get_local_size(0) >> 3) + (lid >> 3);
    const int nb   = ncols / QK_K;
    const int l    = 4*j;
    const int is   = j >> 2;

    float acc = 0.0f;
    if (row < nrows) {
        const int n0 = row * nb;
        __global const uchar * blk = xq + (size_t) n0 * 210;
        for (int i = 0; i < nb; ++i, blk += 210) {
            const uint   odd   = (uint) ((n0 + i) & 1) << 1;
            const uint   sh    = odd << 3;
            const uint   hmask = 0u - (odd >> 1);
            __global const uint  * pa  = (__global const uint *) (blk - odd);
            __global const char  * sc  = (__global const char *) (blk + 192);
            __local  const float * yb  = yl + i*QK_K + l;
            const float d  = vload_half(0, (__global const half *) (blk + 208));
            const uint  h0 = q6_ld_u32(pa, (128 + l)/4, sh, hmask);
            const uint  h1 = q6_ld_u32(pa, (160 + l)/4, sh, hmask);
            float sum = 0.0f;
            sum += q6_group_l(q6_ld_u32(pa, ( 0 + l)/4, sh, hmask), h0, 0, yb +   0, (float) sc[ 0 + is], (float) sc[ 4 + is]);
            sum += q6_group_l(q6_ld_u32(pa, (32 + l)/4, sh, hmask), h0, 1, yb +  32, (float) sc[ 2 + is], (float) sc[ 6 + is]);
            sum += q6_group_l(q6_ld_u32(pa, (64 + l)/4, sh, hmask), h1, 0, yb + 128, (float) sc[ 8 + is], (float) sc[12 + is]);
            sum += q6_group_l(q6_ld_u32(pa, (96 + l)/4, sh, hmask), h1, 1, yb + 160, (float) sc[10 + is], (float) sc[14 + is]);
            acc += d * sum;
        }
    }
    kq_reduce8_store(tmp, lid, j, acc, row, nrows, dst);
}

);

// Kernels of the complete OpenCL backend (ggml_backend_opencl). Every tensor is passed as the
// cl_mem of its backend buffer plus a byte offset, and every stride (nb*) is in bytes, so views at
// any offset work. They must be compiled after dmmv_kq_l8_source (they use its helpers).
std::string backend_ops_source = MULTILINE_QUOTE(

// ggml_add / ggml_mul with broadcasting of src1; dst has the shape of src0; dimension 0 of every
// tensor must be contiguous. Every row (i1, i2, i3) of dst is split into nch chunks of 4 * local size
// values, one work-group per chunk (with one work-group per row, a single row such as the vectors of a
// generated token, 2048-6144 values, ran on one multiprocessor: about 20 us per operation).
__kernel void kernel_add_f32(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                             __global char * dst, const ulong offd, const int ne00,
                             const ulong nb01, const ulong nb02, const ulong nb03,
                             const int ne10, const int ne11, const int ne12, const int ne13,
                             const ulong nb11, const ulong nb12, const ulong nb13,
                             const ulong nb1, const ulong nb2, const ulong nb3, const int nch) {
    const int i1 = get_group_id(0) / nch;
    const int ch = get_group_id(0) % nch;
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);
    const int ls = get_local_size(0);
    const int i0e = min(ne00, (ch + 1) * 4*ls);
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    __global const float * y = (__global const float *) (src1 + off1 + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    if (ne10 == ne00) {
        for (int i0 = ch * 4*ls + get_local_id(0); i0 < i0e; i0 += ls) {
            d[i0] = x[i0] + y[i0];
        }
    } else {
        for (int i0 = ch * 4*ls + get_local_id(0); i0 < i0e; i0 += ls) {
            d[i0] = x[i0] + y[i0 % ne10];
        }
    }
}

__kernel void kernel_mul_f32(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                             __global char * dst, const ulong offd, const int ne00,
                             const ulong nb01, const ulong nb02, const ulong nb03,
                             const int ne10, const int ne11, const int ne12, const int ne13,
                             const ulong nb11, const ulong nb12, const ulong nb13,
                             const ulong nb1, const ulong nb2, const ulong nb3, const int nch) {
    const int i1 = get_group_id(0) / nch;
    const int ch = get_group_id(0) % nch;
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);
    const int ls = get_local_size(0);
    const int i0e = min(ne00, (ch + 1) * 4*ls);
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    __global const float * y = (__global const float *) (src1 + off1 + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    if (ne10 == ne00) {
        for (int i0 = ch * 4*ls + get_local_id(0); i0 < i0e; i0 += ls) {
            d[i0] = x[i0] * y[i0];
        }
    } else {
        for (int i0 = ch * 4*ls + get_local_id(0); i0 < i0e; i0 += ls) {
            d[i0] = x[i0] * y[i0 % ne10];
        }
    }
}

// contiguous tensors only
__kernel void kernel_scale_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                               const float s, const float b, const uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        ((__global float *) (dst + offd))[i] = ((__global const float *) (src0 + off0))[i] * s + b;
    }
}

// contiguous tensors only
__kernel void kernel_silu_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                              const uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        const float x = ((__global const float *) (src0 + off0))[i];
        // x * sigmoid(x) with exp(-|x|) <= 1, so that nothing overflows: the program is built with
        // -cl-finite-math-only, and with an infinity (exp(-x) for x < -88) the result is undefined
        // (PoCL gives NaN where the NVIDIA compiler happens to give -0)
        const float e = exp(-fabs(x));
        ((__global float *) (dst + offd))[i] = x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e);
    }
}

// one work-group (power-of-two size) per row; dimension 0 contiguous
__kernel void kernel_rms_norm_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                                  const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                                  const ulong nb1, const ulong nb2, const ulong nb3, const float eps,
                                  __local float * sum) {
    const int i1  = get_group_id(0);
    const int i2  = get_group_id(1);
    const int i3  = get_group_id(2);
    const int lid = get_local_id(0);
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    float s = 0.0f;
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        s += x[i0] * x[i0];
    }
    sum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (lid < k) {
            sum[lid] += sum[lid + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / sqrt(sum[0] / ne00 + eps);
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        d[i0] = x[i0] * scale;
    }
}

// ggml_cpy / ggml_cont / ggml_dup between F32, F16 and I32 (src_t, dst_t: 0 F32, 1 F16, 2 I32; F32 -> I32
// truncates like the CPU) with any strides; src0 and dst may have different shapes with the same number of
// elements (element i of src0 goes to element i of dst)
__kernel void kernel_cpy(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                         const int ne00, const int ne01, const int ne02,
                         const ulong nb00, const ulong nb01, const ulong nb02, const ulong nb03,
                         const int ne0, const int ne1, const int ne2,
                         const ulong nb0, const ulong nb1, const ulong nb2, const ulong nb3,
                         const int src_t, const int dst_t, const uint n) {
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uint r = i;
    const uint i00 = r % (uint) ne00; r /= (uint) ne00;
    const uint i01 = r % (uint) ne01; r /= (uint) ne01;
    const uint i02 = r % (uint) ne02;
    const uint i03 = r / (uint) ne02;
    r = i;
    const uint i0 = r % (uint) ne0; r /= (uint) ne0;
    const uint i1 = r % (uint) ne1; r /= (uint) ne1;
    const uint i2 = r % (uint) ne2;
    const uint i3 = r / (uint) ne2;
    __global const char * s = src0 + off0 + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
    __global       char * d = dst  + offd + i0*nb0   + i1*nb1   + i2*nb2   + i3*nb3;
    if (src_t == 2 && dst_t == 2) {
        *((__global int *) d) = *((__global const int *) s);
        return;
    }
    const float v = src_t == 2 ? (float) *((__global const int *) s) :
                    src_t == 1 ? vload_half(0, (__global const half *) s) : *((__global const float *) s);
    if (dst_t == 1) {
        vstore_half(v, 0, (__global half *) d);
    } else if (dst_t == 2) {
        *((__global int *) d) = (int) v;
    } else {
        *((__global float *) d) = v;
    }
}

// transposition of F16 matrices (V of the decomposed flash attention): src [ne0, ne1, ne2] (rows of ne0 halfs, strides
// nb1 and nb2) -> dst [ne1, ne0, ne2] contiguous, in tiles of 16x16 through local memory so that the reads and the
// writes are contiguous. The halfs are copied as ushort.
__kernel void kernel_transpose_f16(__global const char * src, const ulong offs, const ulong nb1, const ulong nb2,
                                   __global char * dst, const ulong offd, const int ne0, const int ne1) {
    __local ushort tile[16][17];
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int x0 = get_group_id(0)*16;
    const int y0 = get_group_id(1)*16;
    const int i2 = get_group_id(2);
    if (x0 + lx < ne0 && y0 + ly < ne1) {
        tile[ly][lx] = ((__global const ushort *) (src + offs + (ulong) (y0 + ly)*nb1 + (ulong) i2*nb2))[x0 + lx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x0 + ly < ne0 && y0 + lx < ne1) {
        ((__global ushort *) (dst + offd))[((ulong) i2*ne0 + x0 + ly)*ne1 + y0 + lx] = tile[lx][ly];
    }
}

// ggml_arange (F32): dst[i] = start + step*i
__kernel void kernel_arange_f32(__global char * dst, const ulong offd, const float start, const float step, const uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        ((__global float *) (dst + offd))[i] = start + step * (float) i;
    }
}

// ggml_get_rows with F32 or F16 src0 and I32 row indices; dst is F32. Every row of dst is split into nch chunks of
// csz values, one work-group per chunk (the recurrent states of Qwen3.5 are rows of up to 262144 values: copied by a
// single work-group, they took 1.1 ms each on the GT 720M)
__kernel void kernel_get_rows(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                              __global char * dst, const ulong offd,
                              const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                              const ulong nb10, const ulong nb11, const ulong nb12,
                              const ulong nb1, const ulong nb2, const ulong nb3, const int src_f16,
                              const int nch, const int csz) {
    const int i10 = get_group_id(0) / nch;
    const int ch  = get_group_id(0) % nch;
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    __global const char  * s = src0 + off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03;
    __global       float * d = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    const int i0e = min(ne00, (ch + 1)*csz);
    for (int i0 = ch*csz + get_local_id(0); i0 < i0e; i0 += get_local_size(0)) {
        d[i0] = src_f16 ? vload_half(i0, (__global const half *) s) : ((__global const float *) s)[i0];
    }
}

// ggml_soft_max_ext: dst = softmax(src0*scale + slope*mask), mask (F32 or F16, optional) broadcast
// over the rows like the CPU code (mask row = row % ne01). src0 and dst contiguous. One work-group
// (power-of-two size) per row.
__kernel void kernel_soft_max(__global const char * src0, const ulong off0, __global const char * mask, const ulong offm,
                              __global char * dst, const ulong offd,
                              const int ne00, const int ne01, const int ne02, const ulong nb01, const ulong nbm1,
                              const ulong nbm2, const ulong nbm3, const int ne12, const int ne13,
                              const ulong nb1, const int has_mask, const int mask_f16,
                              const float scale, const float max_bias, const float m0, const float m1, const int n_head_log2,
                              __local float * buf) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float * x = (__global const float *) (src0 + off0 + (ulong) row*nb01);
    __global       float * d = (__global       float *) (dst  + offd + (ulong) row*nb1);
    const int i01 = row % ne01;
    const int i02 = (row / ne01) % ne02;
    const int i03 = row / (ne01*ne02);
    __global const char  * mrow = mask + offm + (ulong) i01*nbm1 + (ulong) (i02 % ne12)*nbm2 + (ulong) (i03 % ne13)*nbm3;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const int h = (row / ne01) % ne02;
        slope = h < n_head_log2 ? pow(m0, (float) (h + 1)) : pow(m1, (float) (2*(h - n_head_log2) + 1));
    }

    // pass 1: maximum. The program is built with -cl-finite-math-only, so no infinity may take part in
    // the arithmetic (the result would be undefined on some compilers): the maximum starts from -FLT_MAX,
    // and the masked positions (-INFINITY in the mask of llama.cpp) are detected with a comparison
    // (mask < -1e30, which the compiler cannot remove) and get a weight of 0.
    float mx = -FLT_MAX;
    for (int i = lid; i < ne00; i += lsz) {
        float v = x[i] * scale;
        if (has_mask) {
            const float mv = mask_f16 ? vload_half(i, (__global const half *) mrow) : ((__global const float *) mrow)[i];
            if (mv < -1e30f) {
                continue;
            }
            v += slope * mv;
        }
        mx = max(mx, v);
    }
    buf[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = lsz / 2; k > 0; k >>= 1) {
        if (lid < k) {
            buf[lid] = max(buf[lid], buf[lid + k]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // pass 2: exponentials (stored in dst) and their sum
    float sum = 0.0f;
    for (int i = lid; i < ne00; i += lsz) {
        float v = x[i] * scale;
        float e = 0.0f;
        if (has_mask) {
            const float mv = mask_f16 ? vload_half(i, (__global const half *) mrow) : ((__global const float *) mrow)[i];
            if (mv >= -1e30f) {
                v += slope * mv;
                e = exp(v - mx);
            }
        } else {
            e = exp(v - mx);
        }
        d[i] = e;
        sum += e;
    }
    buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = lsz / 2; k > 0; k >>= 1) {
        if (lid < k) {
            buf[lid] += buf[lid + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // a row masked everywhere (no weight) gives 0, as in flash attention, instead of a division by zero
    const float inv = buf[0] > 0.0f ? 1.0f / buf[0] : 0.0f;

    // pass 3: normalize
    for (int i = lid; i < ne00; i += lsz) {
        d[i] *= inv;
    }
}

// generic ggml_mul_mat with F32 or F16 src0 and F32 src1, any strides, broadcasting of src0 over
// dimensions 2 and 3 (r2 = ne12/ne02, r3 = ne13/ne03). One work-item per element of dst (contiguous
// in dimension 0). Used for the attention products (K*Q and V*softmax).
__kernel void kernel_mul_mat_f(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                               __global char * dst, const ulong offd,
                               const int ne00, const int ne01, const int ne11, const int ne12,
                               const ulong nb00, const ulong nb01, const ulong nb02, const ulong nb03,
                               const ulong nb10, const ulong nb11, const ulong nb12, const ulong nb13,
                               const ulong nb1, const ulong nb2, const ulong nb3,
                               const int r2, const int r3, const int src0_f16) {
    const int i01 = get_global_id(0);
    const int i11 = get_global_id(1);
    const int i1x = get_global_id(2);
    if (i01 >= ne01 || i11 >= ne11) {
        return;
    }
    const int i12 = i1x % ne12;
    const int i13 = i1x / ne12;
    __global const char * x = src0 + off0 + i01*nb01 + (i12 / r2)*nb02 + (i13 / r3)*nb03;
    __global const char * y = src1 + off1 + i11*nb11 + i12*nb12 + i13*nb13;
    float acc = 0.0f;
    if (src0_f16) {
        for (int k = 0; k < ne00; ++k) {
            acc += vload_half(0, (__global const half *) (x + k*nb00)) * *((__global const float *) (y + k*nb10));
        }
    } else {
        for (int k = 0; k < ne00; ++k) {
            acc += *((__global const float *) (x + k*nb00)) * *((__global const float *) (y + k*nb10));
        }
    }
    *((__global float *) (dst + offd + i01*4 + i11*nb1 + i12*nb2 + i13*nb3)) = acc;
}

// tiled generic MUL_MAT for batches: src0 F32/F16 and src1 F32, both contiguous in dimension 0 (the
// dimension of the sum), any other strides, broadcasting of src0 over dimensions 2 and 3. A 16x16
// work-group computes a 64x64 tile of dst (rows i01, columns i11), every work-item 4x4 values; the
// sum dimension advances in steps of 16 through local memory (rows padded to 65 floats to avoid
// bank conflicts).
__kernel void kernel_mul_mat_f_tiled(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                     __global char * dst, const ulong offd,
                                     const int ne00, const int ne01, const int ne11, const int ne12,
                                     const ulong nb01, const ulong nb02, const ulong nb03,
                                     const ulong nb11, const ulong nb12, const ulong nb13,
                                     const ulong nb1, const ulong nb2, const ulong nb3,
                                     const int r2, const int r3, const int src0_f16) {
    __local float As[16][65];
    __local float Bs[16][65];

    const int tx  = get_local_id(0);
    const int ty  = get_local_id(1);
    const int tid = ty*16 + tx;
    const int m0  = get_group_id(0) * 64;
    const int n0  = get_group_id(1) * 64;
    const int i1x = get_group_id(2);
    const int i12 = i1x % ne12;
    const int i13 = i1x / ne12;

    __global const char * x = src0 + off0 + (i12 / r2)*nb02 + (i13 / r3)*nb03;
    __global const char * y = src1 + off1 + i12*nb12 + i13*nb13;

    float acc[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < ne00; k0 += 16) {
        // 16 (k) x 64 values of src0 and of src1, 4 per work-item each; consecutive work-items read
        // consecutive k, which are contiguous in memory
        for (int e = 0; e < 4; ++e) {
            const int li = tid + 256*e;
            const int kk = li & 15;
            const int mm = li >> 4;
            const int k  = k0 + kk;
            const int m  = m0 + mm;
            const int n  = n0 + mm;
            float va = 0.0f;
            if (k < ne00 && m < ne01) {
                __global const char * px = x + (ulong) m*nb01;
                va = src0_f16 ? vload_half(k, (__global const half *) px) : ((__global const float *) px)[k];
            }
            As[kk][mm] = va;
            float vb = 0.0f;
            if (k < ne00 && n < ne11) {
                vb = ((__global const float *) (y + (ulong) n*nb11))[k];
            }
            Bs[kk][mm] = vb;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 16; ++kk) {
            float a[4];
            float b[4];
            for (int r = 0; r < 4; ++r) {
                a[r] = As[kk][tx + 16*r];
            }
            for (int c = 0; c < 4; ++c) {
                b[c] = Bs[kk][ty + 16*c];
            }
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    acc[r][c] += a[r]*b[c];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int c = 0; c < 4; ++c) {
        const int n = n0 + ty + 16*c;
        if (n < ne11) {
            __global float * d = (__global float *) (dst + offd + (ulong) n*nb1 + i12*nb2 + i13*nb3);
            for (int r = 0; r < 4; ++r) {
                const int m = m0 + tx + 16*r;
                if (m < ne01) {
                    d[m] = acc[r][c];
                }
            }
        }
    }
}

// dequantization of one 256-weight super-block by 8 work-items (lane j writes 32 weights, starting at
// y = block output + 4*j); shared by the whole-matrix dequantization and the quantized GET_ROWS
inline void dq_block_q4_K(__global const uchar * blk, const int j, __global float * y) {
    const float2 dm = vload_half2(0, (__global const half *) blk);
    float4 sc_a, m_a, sc_b, m_b;
    kq_scales_mins(blk, &sc_a, &m_a, &sc_b, &m_b);
    const float sc[8] = { sc_a.x, sc_a.y, sc_a.z, sc_a.w, sc_b.x, sc_b.y, sc_b.z, sc_b.w };
    const float mn[8] = { m_a.x,  m_a.y,  m_a.z,  m_a.w,  m_b.x,  m_b.y,  m_b.z,  m_b.w  };
    __global const uint * qs = (__global const uint *) (blk + 16);
    for (int k = 0; k < 4; ++k) {
        const uint w = qs[8*k + j];
        vstore4(dm.x*sc[2*k]     * convert_float4(as_uchar4( w       & 0x0F0F0F0Fu)) - dm.y*mn[2*k],     0, y + 64*k);
        vstore4(dm.x*sc[2*k + 1] * convert_float4(as_uchar4((w >> 4) & 0x0F0F0F0Fu)) - dm.y*mn[2*k + 1], 0, y + 64*k + 32);
    }
}

inline void dq_block_q5_K(__global const uchar * blk, const int j, __global float * y) {
    const float2 dm = vload_half2(0, (__global const half *) blk);
    float4 sc_a, m_a, sc_b, m_b;
    kq_scales_mins(blk, &sc_a, &m_a, &sc_b, &m_b);
    const float sc[8] = { sc_a.x, sc_a.y, sc_a.z, sc_a.w, sc_b.x, sc_b.y, sc_b.z, sc_b.w };
    const float mn[8] = { m_a.x,  m_a.y,  m_a.z,  m_a.w,  m_b.x,  m_b.y,  m_b.z,  m_b.w  };
    const uint hq = ((__global const uint *) (blk + 16))[j];
    __global const uint * qs = (__global const uint *) (blk + 48);
    for (int k = 0; k < 4; ++k) {
        const uint w = qs[8*k + j];
        const uint lo = ( w       & 0x0F0F0F0Fu) | (((hq >> (2*k))     & 0x01010101u) << 4);
        const uint hi = ((w >> 4) & 0x0F0F0F0Fu) | (((hq >> (2*k + 1)) & 0x01010101u) << 4);
        vstore4(dm.x*sc[2*k]     * convert_float4(as_uchar4(lo)) - dm.y*mn[2*k],     0, y + 64*k);
        vstore4(dm.x*sc[2*k + 1] * convert_float4(as_uchar4(hi)) - dm.y*mn[2*k + 1], 0, y + 64*k + 32);
    }
}

// odd: 2 if the block starts at an address = 2 (mod 4), else 0 (see dequantize_mul_mat_vec_q6_K_l8)
inline void dq_block_q6_K(__global const uchar * blk, const uint odd, const int j, __global float * y) {
    const int  l     = 4*j;
    const int  is    = j >> 2;
    const uint sh    = odd << 3;
    const uint hmask = 0u - (odd >> 1);
    __global const uint * pa = (__global const uint *) (blk - odd);
    __global const char * sc = (__global const char *) (blk + 192);
    const float d = vload_half(0, (__global const half *) (blk + 208));
    for (int h = 0; h < 2; ++h) {
        const uint hq = q6_ld_u32(pa, (128 + 32*h + l)/4, sh, hmask);
        for (int sel = 0; sel < 2; ++sel) {
            const uint w = q6_ld_u32(pa, (64*h + 32*sel + l)/4, sh, hmask);
            const float4 qlo = convert_float4(as_uchar4(( w       & 0x0F0F0F0Fu) | (((hq >> (2*sel))     & 0x03030303u) << 4))) - 32.0f;
            const float4 qhi = convert_float4(as_uchar4(((w >> 4) & 0x0F0F0F0Fu) | (((hq >> (4 + 2*sel)) & 0x03030303u) << 4))) - 32.0f;
            vstore4(d * (float) sc[8*h + is + 2*sel]     * qlo, 0, y + 128*h + 32*sel);
            vstore4(d * (float) sc[8*h + is + 2*sel + 4] * qhi, 0, y + 128*h + 32*sel + 64);
        }
    }
}

// GET_ROWS with Q4_K / Q5_K / Q6_K rows (token embeddings): one work-group per row of dst, 8 lanes
// per super-block. nb = super-blocks per row.
__kernel void kernel_get_rows_q4_K(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                   __global char * dst, const ulong offd,
                                   const int nb, const ulong nb01, const ulong nb02, const ulong nb03,
                                   const ulong nb10, const ulong nb11, const ulong nb12,
                                   const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    __global const uchar * row = (__global const uchar *) (src0 + off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03);
    __global       float * d   = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    const int j = get_local_id(0) & 7;
    for (int b = get_local_id(0) >> 3; b < nb; b += get_local_size(0) >> 3) {
        dq_block_q4_K(row + (size_t) b * 144, j, d + (size_t) b * QK_K + 4*j);
    }
}

__kernel void kernel_get_rows_q5_K(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                   __global char * dst, const ulong offd,
                                   const int nb, const ulong nb01, const ulong nb02, const ulong nb03,
                                   const ulong nb10, const ulong nb11, const ulong nb12,
                                   const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    __global const uchar * row = (__global const uchar *) (src0 + off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03);
    __global       float * d   = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    const int j = get_local_id(0) & 7;
    for (int b = get_local_id(0) >> 3; b < nb; b += get_local_size(0) >> 3) {
        dq_block_q5_K(row + (size_t) b * 176, j, d + (size_t) b * QK_K + 4*j);
    }
}

// the alignment of every block is taken from its byte offset (the buffer start is 4-byte aligned)
__kernel void kernel_get_rows_q6_K(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                   __global char * dst, const ulong offd,
                                   const int nb, const ulong nb01, const ulong nb02, const ulong nb03,
                                   const ulong nb10, const ulong nb11, const ulong nb12,
                                   const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    const ulong roff = off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03;
    __global float * d = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    const int j = get_local_id(0) & 7;
    for (int b = get_local_id(0) >> 3; b < nb; b += get_local_size(0) >> 3) {
        const ulong boff = roff + (ulong) b * 210;
        dq_block_q6_K((__global const uchar *) (src0 + boff), (uint) (boff & 2), j, d + (size_t) b * QK_K + 4*j);
    }
}

// dequantization of whole Q4_K / Q5_K / Q6_K matrices to F32 (for the GEMM of batches);
// 8 work-items per 256-weight super-block, 32 weights each
__kernel void kernel_dequant_q4_K(__global const uchar * xq_base, const ulong xoff, __global char * dst_base, const ulong doff,
                                  const uint nblocks) {
    const uint gid = get_global_id(0);
    const uint b   = gid >> 3;
    const int  j   = gid & 7;
    if (b >= nblocks) {
        return;
    }
    dq_block_q4_K(xq_base + xoff + (size_t) b * 144, j, (__global float *) (dst_base + doff) + (size_t) b * QK_K + 4*j);
}

__kernel void kernel_dequant_q5_K(__global const uchar * xq_base, const ulong xoff, __global char * dst_base, const ulong doff,
                                  const uint nblocks) {
    const uint gid = get_global_id(0);
    const uint b   = gid >> 3;
    const int  j   = gid & 7;
    if (b >= nblocks) {
        return;
    }
    dq_block_q5_K(xq_base + xoff + (size_t) b * 176, j, (__global float *) (dst_base + doff) + (size_t) b * QK_K + 4*j);
}

// xoff must be a multiple of 4 (see dequantize_mul_mat_vec_q6_K_l8 for the alignment trick)
__kernel void kernel_dequant_q6_K(__global const uchar * xq_base, const ulong xoff, __global char * dst_base, const ulong doff,
                                  const uint nblocks) {
    const uint gid = get_global_id(0);
    const uint b   = gid >> 3;
    const int  j   = gid & 7;
    if (b >= nblocks) {
        return;
    }
    dq_block_q6_K(xq_base + xoff + (size_t) b * 210, (b & 1) << 1, j, (__global float *) (dst_base + doff) + (size_t) b * QK_K + 4*j);
}

// unary ops without overflow (the program is built with -cl-finite-math-only: no infinity may appear)
__kernel void kernel_sigmoid_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                                 const uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        const float x = ((__global const float *) (src0 + off0))[i];
        const float e = exp(-fabs(x));
        ((__global float *) (dst + offd))[i] = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    }
}

__kernel void kernel_softplus_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                                  const uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        const float x = ((__global const float *) (src0 + off0))[i];
        ((__global float *) (dst + offd))[i] = x > 20.0f ? x : log1p(exp(x));
    }
}

// CONCAT (F32) along dim, with any strides (Qwen3.5 concatenates a transposed view); one work-item per
// element of dst; ne0d = size of src0 along dim
__kernel void kernel_concat_f32(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                __global char * dst, const ulong offd,
                                const int ne0, const int ne1, const int ne2,
                                const ulong nb00, const ulong nb01, const ulong nb02, const ulong nb03,
                                const ulong nb10, const ulong nb11, const ulong nb12, const ulong nb13,
                                const ulong nb0, const ulong nb1, const ulong nb2, const ulong nb3,
                                const int dim, const int ne0d, const uint n) {
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int idx[4];
    uint r = i;
    idx[0] = r % ne0; r /= ne0;
    idx[1] = r % ne1; r /= ne1;
    idx[2] = r % ne2;
    idx[3] = r / ne2;
    const ulong od = offd + idx[0]*nb0 + idx[1]*nb1 + idx[2]*nb2 + idx[3]*nb3;
    float x;
    if (idx[dim] < ne0d) {
        x = *((__global const float *) (src0 + off0 + idx[0]*nb00 + idx[1]*nb01 + idx[2]*nb02 + idx[3]*nb03));
    } else {
        idx[dim] -= ne0d;
        x = *((__global const float *) (src1 + off1 + idx[0]*nb10 + idx[1]*nb11 + idx[2]*nb12 + idx[3]*nb13));
    }
    *((__global float *) (dst + od)) = x;
}

// SSM_CONV1D (ggml_ssm_conv1d): dst[i1, t, s] = sum_k sx[t + k, i1, s] * c[k, i1]; one work-item per
// channel, token and sequence
__kernel void kernel_ssm_conv1d_f32(__global const char * sx, const ulong offx, const ulong nbx1, const ulong nbx2,
                                    __global const char * c, const ulong offc, const ulong nbc1,
                                    __global char * dst, const ulong offd, const ulong nb1, const ulong nb2,
                                    const int d_conv, const int nr) {
    const int i1 = get_global_id(0);
    const int t  = get_global_id(1);
    const int s  = get_global_id(2);
    if (i1 >= nr) {
        return;
    }
    __global const float * x = (__global const float *) (sx + offx + i1*nbx1 + s*nbx2) + t;
    __global const float * w = (__global const float *) (c + offc + i1*nbc1);
    float sum = 0.0f;
    for (int k = 0; k < d_conv; ++k) {
        sum += x[k] * w[k];
    }
    ((__global float *) (dst + offd + t*nb1 + s*nb2))[i1] = sum;
}

// GATED_DELTA_NET (ggml_gated_delta_net) for S_v = 128 (all Qwen3.5 models). The columns j of the state
// S (S[i][j], stored with i contiguous) evolve independently, so every column is held in registers by 16
// consecutive work-items (8 values each, rows lane + 16*e); a work-group of lsz work-items holds lsz/16
// columns of one head and one sequence. For every token: q and k are staged in local memory, and the
// two sums over i (S^T k and S^T q) are reduced in local memory (3 barriers per token).
//   q, k: [128, H_k, n_t, n_s] (rows contiguous), head h uses q/k head h % H_k; v: [128, H, n_t, n_s]
//   g, beta: contiguous [1, H, n_t, n_s]; state: contiguous [128, 128, H, n_s]
//   dst: o [128, H, n_t, n_s] followed by n_snap states [128, 128, H, n_s] (snap_stride floats apart):
//   snapshot i = state after token n_t - 1 - i (i = 0: the final state; see ggml_gated_delta_net_ext)
//   use_rid: the state of sequence s is row rid[s] (nbr bytes apart) of st, the recurrent state tensor (the
//   GET_ROWS that gathers it is fused, see ggml_cl_be_gdn_rows)
__kernel void kernel_gated_delta_net_f32(
        __global const char * q, const ulong offq, const ulong nbq1, const ulong nbq2, const ulong nbq3, const int neq1, const int rq3,
        __global const char * k, const ulong offk, const ulong nbk1, const ulong nbk2, const ulong nbk3, const int nek1, const int rk3,
        __global const char * v, const ulong offv, const ulong nbv1, const ulong nbv2, const ulong nbv3,
        __global const char * g, const ulong offg,
        __global const char * b, const ulong offb,
        __global const char * st, const ulong offs,
        __global const char * rid, const ulong offr, const ulong nbr, const int use_rid,
        __global char * dst, const ulong offd,
        __global char * so, const ulong offso,
        const int H, const int n_t, const int n_snap, const ulong snap_stride,
        __local float * kq, __local float * red) {
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int lane = lid & 15;
    const int cl   = lid >> 4;
    const int j    = get_group_id(0) * (lsz >> 4) + cl;
    const int h    = get_group_id(1);
    const int s    = get_group_id(2);
    const int n_s  = get_num_groups(2);

    const int hq = h % neq1;
    const int hk = h % nek1;
    const int sq = s / rq3;
    const int sk = s / rk3;

    __global const float * st_in  = use_rid ?
        (__global const float *) (st + offs + (ulong) ((__global const int *) (rid + offr))[s]*nbr) + ((size_t) h*128 + j)*128 :
        (__global const float *) (st + offs) + ((size_t) (s*H + h)*128 + j)*128;
    __global       float * o_base = (__global float *) (dst + offd);
    // final state: after o in dst, or straight into the state tensor (so may be st: every work-group reads
    // its columns of the state before writing them)
    __global       float * st_out = (__global float *) (so + offso) + ((size_t) (s*H + h)*128 + j)*128;
    __global const float * gp = (__global const float *) (g + offg) + (size_t) s*H*n_t + h;
    __global const float * bp = (__global const float *) (b + offb) + (size_t) s*H*n_t + h;

    float S[8];
    for (int e = 0; e < 8; ++e) {
        S[e] = st_in[lane + 16*e];
    }

    const float scale = 0.08838834764831845f; // 1/sqrt(128)

    for (int t = 0; t < n_t; ++t) {
        __global const float * kt = (__global const float *) (k + offk + sk*nbk3 + t*nbk2 + hk*nbk1);
        __global const float * qt = (__global const float *) (q + offq + sq*nbq3 + t*nbq2 + hq*nbq1);
        for (int i = lid; i < 256; i += lsz) {
            kq[i] = i < 128 ? kt[i] : qt[i - 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const float decay = exp(gp[(size_t) t*H]);
        const float beta  = bp[(size_t) t*H];
        const float vj    = ((__global const float *) (v + offv + s*nbv3 + t*nbv2 + h*nbv1))[j];

        // S <- decay * S; partial sums of S^T k
        float part = 0.0f;
        for (int e = 0; e < 8; ++e) {
            S[e] *= decay;
            part += S[e] * kq[lane + 16*e];
        }
        red[lid] = part;
        barrier(CLK_LOCAL_MEM_FENCE);

        float skj = 0.0f;
        for (int l = 0; l < 16; ++l) {
            skj += red[(cl << 4) + l];
        }
        const float d = (vj - skj) * beta;

        // S[i][j] += k[i] d; partial sums of S^T q
        float po = 0.0f;
        for (int e = 0; e < 8; ++e) {
            S[e] += kq[lane + 16*e] * d;
            po += S[e] * kq[128 + lane + 16*e];
        }
        // snapshot of the state after this token (the last n_snap - 1 tokens before the final one)
        const int slot = n_t - 1 - t;
        if (slot > 0 && slot < n_snap) {
            for (int e = 0; e < 8; ++e) {
                st_out[(size_t) slot*snap_stride + lane + 16*e] = S[e];
            }
        }
        red[lsz + lid] = po;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            float oj = 0.0f;
            for (int l = 0; l < 16; ++l) {
                oj += red[lsz + (cl << 4) + l];
            }
            o_base[((size_t) (s*n_t + t)*H + h)*128 + j] = oj * scale;
        }
    }

    for (int e = 0; e < 8; ++e) {
        st_out[lane + 16*e] = S[e];
    }
}

);

// GEMM of the dequantized weights (prompt processing), tuned on the GT 720M with
// examples/bench-gemm: dst[n][m] = sum_k A[m][k] * src1[n][k], with A the F32 dequantization buffer
// (M rows of K floats, row stride lda floats), src1 N rows of K floats (row stride ldb floats) and dst
// N rows of M floats (row stride ldc floats). It is compiled once per tile shape, with BM, BN, BK, TM,
// TN and PAD defined before the source: a BM x BN tile of dst per work-group of (BM/TM) x (BN/TN)
// work-items, TM x TN values per work-item (strided by BM/TM and BN/TN, so the local memory reads of a
// warp are consecutive or broadcast), steps of BK along k.
// It is a raw string instead of MULTILINE_QUOTE because it needs #pragma unroll: the NVIDIA compiler
// does not unroll these loops by itself, and unrolling them makes the kernel 10-20% faster. Every
// work-item always loads the same column of k (NTHR % BK == 0) through a pointer that moves by BK;
// the indices are 32-bit; out-of-range rows are clamped instead of branching (their results are never
// stored); the local rows are padded by PAD floats (2: no bank conflicts in the transposed stores).
// Requires K % BK == 0, (BM*BK) % NTHR == 0, (BN*BK) % NTHR == 0 and M*lda, N*ldb, N*ldc < 2^31.
static const char * be_gemm_source = R"CLC(
#define NTHR ((BM/TM) * (BN/TN))
__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void kernel_gemm_f32(__global const float * A, const int lda,
                     __global const char * src1, const ulong off1, const int ldb,
                     __global char * dst, const ulong offd, const int ldc,
                     const int M, const int N, const int K) {
    __local float As[BK][BM + PAD];
    __local float Bs[BK][BN + PAD];

    __global const float * B = (__global const float *) (src1 + off1);
    __global float *       C = (__global float *) (dst + offd);

    const int tx  = get_local_id(0);
    const int ty  = get_local_id(1);
    const int tid = ty * (BM/TM) + tx;
    const int m0  = get_group_id(0) * BM;
    const int n0  = get_group_id(1) * BN;
    const int kl  = tid % BK;
    const int rl  = tid / BK;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    __global const float * pa = A + kl;
    __global const float * pb = B + kl;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int e = 0; e < (BM*BK)/NTHR; ++e) {
            const int r = rl + e*(NTHR/BK);
            As[kl][r] = pa[min(m0 + r, M - 1)*lda];
        }
        #pragma unroll
        for (int e = 0; e < (BN*BK)/NTHR; ++e) {
            const int r = rl + e*(NTHR/BK);
            Bs[kl][r] = pb[min(n0 + r, N - 1)*ldb];
        }
        pa += BK;
        pb += BK;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM];
            float b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][tx + i*(BM/TM)];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][ty + j*(BN/TN)];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + ty + j*(BN/TN);
        if (n < N) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int m = m0 + tx + i*(BM/TM);
                if (m < M) {
                    C[n*ldc + m] = acc[i][j];
                }
            }
        }
    }
}
)CLC";

// Q6_K weights in the "q6r" layout of the OpenCL buffers (see ggml_cl_q6r): every row of nb super-blocks
// is stored as ql' [nb][2][8][8 bytes], qh' [nb][8][8 bytes], sc' [nb][4][4 bytes], d [nb] (half), so it
// keeps the size of a block_q6_K row (210*nb bytes) and every row is still found at row*nb1. Lane j (0..7)
// of a super-block handles half h = j >> 2 and the positions l0 = 8*(j & 3) .. l0 + 7 of that half. Its data
// are three chunks of 8 bytes, A_j = ql[64h + l0 ..] at ql' + 8j, B_j = ql[64h + 32 + l0 ..] at ql' + 64 + 8j
// and C_j = qh[32h + l0 ..] at qh' + 8j, and its 4 scales sc'[4*(2h + is) + g] = scales[8h + is + 2g]
// (is = (j & 3) >> 1, g = 0..3). Its 32 weights are 128h + l0 + 32g + e (e = 0..7): group g takes the low
// nibbles of A (g = 0) or B (g = 1), or the high nibbles of A (g = 2) or B (g = 3), and bits 2g, 2g + 1 of
// C; Q6R_G0..Q6R_G3 build 4 of those 6-bit quants from one 32-bit word of A, B and C.
// Why (examples/bench-dmmv on the GT 720M, BENCH_DMMV_Q6X=1 BENCH_DMMV_Q6R=1): the 210-byte super-blocks
// of block_q6_K are only 2-byte aligned, so every lane rebuilt its words from twice the reads and needed 8
// scales; here every lane reads its quants with 8-byte (or 4-byte) aligned loads and its 4 scales with one
// 32-bit load, and the 8 lanes of a row read contiguous 64-byte segments. Matrix x vector: 8.1-8.5 GB/s
// against 5.1-6.4 GB/s with the standard layout (ceiling of the device: 10.26 GB/s).
static const char * be_q6r_common = R"CLC(
#define Q6R_M4 0x0F0F0F0Fu
#define Q6R_M2 0x30303030u
#define Q6R_G0(a, b, c) (( (a)       & Q6R_M4) | (((c) << 4) & Q6R_M2))
#define Q6R_G1(a, b, c) (( (b)       & Q6R_M4) | (((c) << 2) & Q6R_M2))
#define Q6R_G2(a, b, c) ((((a) >> 4) & Q6R_M4) | ( (c)       & Q6R_M2))
#define Q6R_G3(a, b, c) ((((b) >> 4) & Q6R_M4) | (((c) >> 2) & Q6R_M2))
)CLC";

static const char * be_q6r_source = R"CLC(
// dequantization of super-block i of a q6r row (row start rb, nb super-blocks) by its lane j; y = output
// of the super-block
inline void dq_block_q6r(__global const uchar * rb, const int nb, const int i, const int j, __global float * y) {
    const int   h  = j >> 2;
    const int   p  = j & 3;
    const uint2 a  = vload2(0, (__global const uint *) (rb + (size_t) i*128 + 8*j));
    const uint2 b  = vload2(0, (__global const uint *) (rb + (size_t) i*128 + 64 + 8*j));
    const uint2 c  = vload2(0, (__global const uint *) (rb + (size_t) nb*128 + (size_t) i*64 + 8*j));
    const char4 s  = as_char4(((__global const uint *) (rb + (size_t) nb*192 + (size_t) i*16))[2*h + (p >> 1)]);
    const float d  = vload_half(i, (__global const half *) (rb + (size_t) nb*208));
    const float d0 = d * (float) s.x;
    const float d1 = d * (float) s.y;
    const float d2 = d * (float) s.z;
    const float d3 = d * (float) s.w;
    __global float * yo = y + 128*h + 8*p;
    vstore4(d0 * (convert_float4(as_uchar4(Q6R_G0(a.x, b.x, c.x))) - 32.0f), 0, yo +   0);
    vstore4(d0 * (convert_float4(as_uchar4(Q6R_G0(a.y, b.y, c.y))) - 32.0f), 0, yo +   4);
    vstore4(d1 * (convert_float4(as_uchar4(Q6R_G1(a.x, b.x, c.x))) - 32.0f), 0, yo +  32);
    vstore4(d1 * (convert_float4(as_uchar4(Q6R_G1(a.y, b.y, c.y))) - 32.0f), 0, yo +  36);
    vstore4(d2 * (convert_float4(as_uchar4(Q6R_G2(a.x, b.x, c.x))) - 32.0f), 0, yo +  64);
    vstore4(d2 * (convert_float4(as_uchar4(Q6R_G2(a.y, b.y, c.y))) - 32.0f), 0, yo +  68);
    vstore4(d3 * (convert_float4(as_uchar4(Q6R_G3(a.x, b.x, c.x))) - 32.0f), 0, yo +  96);
    vstore4(d3 * (convert_float4(as_uchar4(Q6R_G3(a.y, b.y, c.y))) - 32.0f), 0, yo + 100);
}

// GET_ROWS with q6r rows (token embeddings): one work-group per row of dst, 8 lanes per super-block
__kernel void kernel_get_rows_q6r(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                  __global char * dst, const ulong offd,
                                  const int nb, const ulong nb01, const ulong nb02, const ulong nb03,
                                  const ulong nb10, const ulong nb11, const ulong nb12,
                                  const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    __global const uchar * row = (__global const uchar *) (src0 + off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03);
    __global       float * d   = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    const int j = get_local_id(0) & 7;
    for (int b = get_local_id(0) >> 3; b < nb; b += get_local_size(0) >> 3) {
        dq_block_q6r(row, nb, b, j, d + (size_t) b * QK_K);
    }
}

// dequantization of a whole q6r matrix (nblocks super-blocks, nb per row) to F32, for the GEMM of batches
__kernel void kernel_dequant_q6r(__global const uchar * xq_base, const ulong xoff, __global char * dst_base, const ulong doff,
                                 const uint nblocks, const int nb) {
    const uint gid = get_global_id(0);
    const uint b   = gid >> 3;
    const int  j   = gid & 7;
    if (b >= nblocks) {
        return;
    }
    const uint r = b / (uint) nb;
    const int  i = (int) (b - r * (uint) nb);
    dq_block_q6r(xq_base + xoff + (size_t) r * nb * 210, nb, i, j, (__global float *) (dst_base + doff) + (size_t) b * QK_K);
}

inline float q6r_dot(const uint q, const float4 y) {
    const float4 f = convert_float4(as_uchar4(q));
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

// y offset in local memory: 4 floats of padding every 128, so that the two halves of a super-block (read
// by lanes 0-3 and 4-7 with 128-bit loads) do not use the same banks
#define Q6R_YOFF(c) ((c) + (((c) >> 7) << 2))
#define Q6R_LDY(p, o) (((__local const float4 *) ((p) + (o)))[0])

// group g of one super-block: sum += scale * (dot(quants, y) - 32 * (sum of those 8 values of y))
#define Q6R_GRP(G, o, yg, s) \
    sum = fma(s, fma(-32.0f, yg, q6r_dot(G(a.x, b.x, c.x), Q6R_LDY(yi, o)) + q6r_dot(G(a.y, b.y, c.y), Q6R_LDY(yi, (o) + 4))), sum)

// matrix x vector with q6r weights: 8 lanes per row, (local size / 8) rows per work-group. y is staged in
// local memory in chunks of ych values (a multiple of 256: the local memory does not depend on the row
// length), together with the sums of every 8 values of y (for the -32 offset of the quants), reordered so
// that the 4 sums of a lane are one float4. v2: 8-byte loads of the quants (rows and xq 8-byte aligned).
// Tuned on the GT 720M with examples/bench-dmmv (work-groups of 256, chunks of 2048: 38 registers, 3
// work-groups per multiprocessor; unrolling the loop takes 53 registers and is slower).
inline void dmmv_q6r(__global const uchar * xq, __global const float * y, __global float * dst,
                     const int ncols, const int nrows, const int ych,
                     __local float * tmp, __local float4 * yl4, __local float4 * ysl4, const int v2) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int h   = j >> 2;
    const int p   = j & 3;
    const int nb  = ncols / QK_K;
    const int row = get_group_id(0) * (lsz >> 3) + (lid >> 3);
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;

    __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 210;
    __global const uchar * pq = rb + 8*j;
    __global const uchar * ph = rb + (size_t) nb*128 + 8*j;
    __global const uint  * ps = (__global const uint *) (rb + (size_t) nb*192) + 2*h + (p >> 1);
    __global const half  * pd = (__global const half *) (rb + (size_t) nb*208);

    float acc = 0.0f;
    for (int k0 = 0; k0 < ncols; k0 += ych) {
        const int len = min(ych, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int e = lid; e < len/8; e += lsz) {
            const float4 u = vload4(0, y + k0 + 8*e);
            const float4 w = vload4(0, y + k0 + 8*e + 4);
            vstore4(u, 0, yl + Q6R_YOFF(8*e));
            vstore4(w, 0, yl + Q6R_YOFF(8*e) + 4);
            ysl[(e & ~15) | ((e & 3) << 2) | ((e >> 2) & 3)] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nbc = len / QK_K;
        for (int i = 0; i < nbc; ++i) {
            uint2 a;
            uint2 b;
            uint2 c;
            if (v2) {
                a = ((__global const uint2 *) pq)[0];
                b = ((__global const uint2 *) (pq + 64))[0];
                c = ((__global const uint2 *) ph)[0];
            } else {
                a = vload2(0, (__global const uint *) pq);
                b = vload2(0, (__global const uint *) (pq + 64));
                c = vload2(0, (__global const uint *) ph);
            }
            const char4  s  = as_char4(*ps);
            const float  d  = vload_half(0, pd);
            const float4 yg = ysl4[8*i + 4*h + p];
            __local const float * yi = yl + Q6R_YOFF(i*QK_K + 128*h + 8*p);
            float sum = 0.0f;
            Q6R_GRP(Q6R_G0,  0, yg.x, (float) s.x);
            Q6R_GRP(Q6R_G1, 32, yg.y, (float) s.y);
            Q6R_GRP(Q6R_G2, 64, yg.z, (float) s.z);
            Q6R_GRP(Q6R_G3, 96, yg.w, (float) s.w);
            acc = fma(d, sum, acc);
            pq += 128;
            ph += 64;
            ps += 4;
            pd += 1;
        }
    }

    tmp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 4; s > 0; s >>= 1) {
        if (j < s) {
            tmp[lid] += tmp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0 && row < nrows) {
        dst[row] = tmp[lid];
    }
}

// local memory: tmp (local size floats), yl4 (ych + ych/32 floats), ysl4 (ych/8 floats)
__kernel void kernel_dmmv_q6r(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff,
                              __global char * dst_base, const ulong doff, const int ncols, const int nrows, const int ych,
                              __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    dmmv_q6r(xq_base + xoff, (__global const float *) (y_base + yoff), (__global float *) (dst_base + doff),
             ncols, nrows, ych, tmp, yl4, ysl4, 0);
}

__kernel void kernel_dmmv_q6r_v2(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff,
                                 __global char * dst_base, const ulong doff, const int ncols, const int nrows, const int ych,
                                 __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    dmmv_q6r(xq_base + xoff, (__global const float *) (y_base + yoff), (__global float *) (dst_base + doff),
             ncols, nrows, ych, tmp, yl4, ysl4, 1);
}
)CLC";

// Matrix x vector for Q4_K and Q5_K (standard block layout) with y in local memory, tuned on the GT 720M
// with examples/bench-dmmv (BENCH_DMMV_Q4AB=1, interleaved measurements after warming up the GPU):
//   - same lane layout as dmmv_kq_l8_source (lane j: weights 4j..4j+3 of every chunk of 32);
//   - y is staged in local memory in chunks of ych values together with the sums of every 8 values;
//   - the min term of a super-block, dmin * sum_q m_q * (sum of y over chunk q), is split among the
//     lanes: lane j adds the one of chunk j, so it decodes only its own min (9 conversions instead of 16
//     and no partial sums of y; 204 instructions per super-block and lane instead of 239);
//   - with y in one chunk (ncols <= ych), every work-group handles T groups of (local size / 8) rows one
//     after another, so the staging of y and its barrier are paid once per T groups (T > 1 needs one chunk).
// Q4_K: 2-12% faster than dequantize_mul_mat_vec_q4_K_l8 (7.5-8.0 GB/s against 6.4-7.5).
static const char * be_q45_source = R"CLC(
inline float q45_dot(const uint q, const float4 y) {
    const float4 f = convert_float4(as_uchar4(q));
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

// y[k0 .. k0 + len) -> yl, sums of every 8 values -> ysl
inline void q45_stage(__global const float * y, const int k0, const int len, __local float * yl, __local float * ysl,
                      const int lid, const int lsz) {
    for (int e = lid; e < len/8; e += lsz) {
        const float4 u = vload4(0, y + k0 + 8*e);
        const float4 w = vload4(0, y + k0 + 8*e + 4);
        vstore4(u, 0, yl + 8*e);
        vstore4(w, 0, yl + 8*e + 4);
        ysl[e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
    }
}

// partial sum of lane j over nbc super-blocks of one row starting at blk (bs = 144: Q4_K, 176: Q5_K),
// with y of those super-blocks staged at yl / ysl4
inline float q45_part(__global const uchar * blk, const int nbc, const int bs, const int j,
                      __local const float * yl, __local const float4 * ysl4) {
    const int jj = j & 3;
    float acc = 0.0f;
    for (int i = 0; i < nbc; ++i, blk += bs) {
        const float2 dm = vload_half2(0, (__global const half *) blk);
        __global const uint * sp = (__global const uint *) (blk + 4);
        const uint s0 = sp[0];
        const uint s1 = sp[1];
        const uint s2 = sp[2];
        const float4 sc_a = convert_float4(as_uchar4(s0 & 0x3F3F3F3Fu));
        const float4 sc_b = convert_float4(as_uchar4((s2 & 0x0F0F0F0Fu) | ((s0 >> 2) & 0x30303030u)));
        const uint   mlo  = (s1 >> (8*jj)) & 63;
        const uint   mhi  = ((s2 >> (8*jj + 4)) & 0xF) | (((s1 >> (8*jj + 6)) & 3) << 4);
        const float  mj   = (float) (j < 4 ? mlo : mhi);
        const float4 ys4  = ysl4[8*i + j];
        const float  ysj  = (ys4.x + ys4.y) + (ys4.z + ys4.w);
        uint hq = 0;
        __global const uint * qs = (__global const uint *) (blk + 16);
        if (bs == 176) {
            hq = qs[j];
            qs = (__global const uint *) (blk + 48);
        }
        const uint w0 = qs[j];
        const uint w1 = qs[j +  8];
        const uint w2 = qs[j + 16];
        const uint w3 = qs[j + 24];
        // 5th bit of the Q5_K quants: bit q of hq for chunk q (hq = 0 for Q4_K)
        const uint h0 = ( hq       & 0x01010101u) << 4;
        const uint h1 = ((hq >> 1) & 0x01010101u) << 4;
        const uint h2 = ((hq >> 2) & 0x01010101u) << 4;
        const uint h3 = ((hq >> 3) & 0x01010101u) << 4;
        const uint h4 = ((hq >> 4) & 0x01010101u) << 4;
        const uint h5 = ((hq >> 5) & 0x01010101u) << 4;
        const uint h6 = ((hq >> 6) & 0x01010101u) << 4;
        const uint h7 = ((hq >> 7) & 0x01010101u) << 4;
        __local const float * yb = yl + i*QK_K + 4*j;
        float sd = 0.0f;
        sd = fma(sc_a.x, q45_dot(( w0       & 0x0F0F0F0Fu) | h0, vload4(0, yb      )), sd);
        sd = fma(sc_a.y, q45_dot(((w0 >> 4) & 0x0F0F0F0Fu) | h1, vload4(0, yb +  32)), sd);
        sd = fma(sc_a.z, q45_dot(( w1       & 0x0F0F0F0Fu) | h2, vload4(0, yb +  64)), sd);
        sd = fma(sc_a.w, q45_dot(((w1 >> 4) & 0x0F0F0F0Fu) | h3, vload4(0, yb +  96)), sd);
        sd = fma(sc_b.x, q45_dot(( w2       & 0x0F0F0F0Fu) | h4, vload4(0, yb + 128)), sd);
        sd = fma(sc_b.y, q45_dot(((w2 >> 4) & 0x0F0F0F0Fu) | h5, vload4(0, yb + 160)), sd);
        sd = fma(sc_b.z, q45_dot(( w3       & 0x0F0F0F0Fu) | h6, vload4(0, yb + 192)), sd);
        sd = fma(sc_b.w, q45_dot(((w3 >> 4) & 0x0F0F0F0Fu) | h7, vload4(0, yb + 224)), sd);
        acc = fma(dm.x, sd, fma(-dm.y * mj, ysj, acc));
    }
    return acc;
}

// reduction of the 8 partial sums of every row of the work-group, one result per row
inline void q45_reduce_store(__local float * tmp, const int lid, const int j, const float acc, const int row,
                             const int nrows, __global float * dst) {
    tmp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 4; s > 0; s >>= 1) {
        if (j < s) {
            tmp[lid] += tmp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0 && row < nrows) {
        dst[row] = tmp[lid];
    }
}

// y in one chunk (ncols <= size of yl4), T groups of rows per work-group.
// Local memory: tmp (local size floats), yl4 (ncols floats), ysl4 (ncols/8 floats).
inline void dmmv_q45_one(__global const uchar * xq, __global const float * y, __global float * dst,
                         const int ncols, const int nrows, const int T, const int bs,
                         __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int nb  = ncols / QK_K;
    q45_stage(y, 0, ncols, (__local float *) yl4, (__local float *) ysl4, lid, lsz);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (lsz >> 3) + (lid >> 3);
        const float acc = q45_part(xq + (size_t) min(row, nrows - 1) * nb * bs, nb, bs, j, (__local const float *) yl4, ysl4);
        q45_reduce_store(tmp, lid, j, acc, row, nrows, dst);
    }
}

// y in chunks of ych values, one group of rows per work-group.
// Local memory: tmp (local size floats), yl4 (ych floats), ysl4 (ych/8 floats).
inline void dmmv_q45_chunks(__global const uchar * xq, __global const float * y, __global float * dst,
                            const int ncols, const int nrows, const int ych, const int bs,
                            __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int nb  = ncols / QK_K;
    const int row = get_group_id(0) * (lsz >> 3) + (lid >> 3);
    __global const uchar * blk = xq + (size_t) min(row, nrows - 1) * nb * bs;
    float acc = 0.0f;
    for (int k0 = 0; k0 < ncols; k0 += ych) {
        const int len = min(ych, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        q45_stage(y, k0, len, (__local float *) yl4, (__local float *) ysl4, lid, lsz);
        barrier(CLK_LOCAL_MEM_FENCE);
        acc += q45_part(blk + (size_t) (k0 / QK_K) * bs, len / QK_K, bs, j, (__local const float *) yl4, ysl4);
    }
    q45_reduce_store(tmp, lid, j, acc, row, nrows, dst);
}

#define Q45_KERNELS(name, bs) \
__kernel void kernel_dmmv_##name##_ly1(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, \
                                      __global char * dst_base, const ulong doff, const int ncols, const int nrows, const int T, \
                                      __local float * tmp, __local float4 * yl4, __local float4 * ysl4) { \
    dmmv_q45_one(xq_base + xoff, (__global const float *) (y_base + yoff), (__global float *) (dst_base + doff), \
                 ncols, nrows, T, bs, tmp, yl4, ysl4); \
} \
__kernel void kernel_dmmv_##name##_lyc(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, \
                                      __global char * dst_base, const ulong doff, const int ncols, const int nrows, const int ych, \
                                      __local float * tmp, __local float4 * yl4, __local float4 * ysl4) { \
    dmmv_q45_chunks(xq_base + xoff, (__global const float *) (y_base + yoff), (__global float *) (dst_base + doff), \
                    ncols, nrows, ych, bs, tmp, yl4, ysl4); \
}

Q45_KERNELS(q4_K, 144)
Q45_KERNELS(q5_K, 176)
)CLC";

// F16/F32 matrix x vectors for the attention of a few tokens (KQ = K x Q and KQV = V x softmax(KQ) while
// generating): src0 rows contiguous (any nb01, nb02, nb03), src1 with any strides and ne11 < 16, the
// broadcasting of ggml (i02 = i12 / r2, i03 = i13 / r3). lpr lanes per row of src0 (a power of two: 8 for
// short rows such as the 256 values of a head of K, 32 for the long rows of V) read it with coalesced
// 4-element loads, and every row is read once for all the vectors that use it: the r2 heads of Q that
// share a head of K/V (grouped-query attention) and the ne11 tokens, in passes of 4 vectors. The generic
// kernel_mul_mat_f (one work-item per output value) was 25 times slower than the bandwidth of the device
// with a context of 2000 tokens. Work-group: a multiple of lpr work-items (local size / lpr rows).
static const char * be_attn_source = R"CLC(
// ARGMAX of every row of an F32 matrix (rows contiguous): one work-group per row (a power of two of
// work-items), the first index of the maximum as on the CPU. The draft tokens of the multi-token
// prediction are chosen on the GPU with it, so only 4 bytes are read back instead of the logits.
__kernel void kernel_argmax_f32(__global const char * src0, const ulong off0, const ulong nb01, const int ne00,
                                __global char * dst, const ulong offd,
                                __local float * rv, __local int * ri) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float * x = (__global const float *) (src0 + off0 + (ulong) row*nb01);
    float mv = -FLT_MAX;
    int   mi = ne00;
    for (int i = lid; i < ne00; i += lsz) {
        const float v = x[i];
        if (v > mv) {
            mv = v;
            mi = i;
        }
    }
    rv[lid] = mv;
    ri[lid] = mi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s) {
            const float ov = rv[lid + s];
            const int   oi = ri[lid + s];
            if (ov > rv[lid] || (ov == rv[lid] && oi < ri[lid])) {
                rv[lid] = ov;
                ri[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        ((__global int *) (dst + offd))[row] = ri[0];
    }
}

// ssm_conv1d for batches (n_t >= 16): work-groups of 16 x 16 handle 16 channels x 16 time steps. The inputs
// of the 16 channels (rows of sx, contiguous in time) are first loaded into local memory with the lanes
// along time (coalesced), then every lane computes one output with the lanes along the channel, so the
// stores to dst (channel-fastest) are coalesced too. kernel_ssm_conv1d_f32 (lanes along the channel: every
// load in a different row of sx) took 1.4 ms per batch of 512 tokens of Qwen3.5-2B. xt: 16*(16 + d_conv - 1).
__kernel void kernel_ssm_conv1d_tiled(__global const char * sx, const ulong offx, const ulong nbx1, const ulong nbx2,
                                      __global const char * c, const ulong offc, const ulong nbc1,
                                      __global char * dst, const ulong offd, const ulong nb1, const ulong nb2,
                                      const int d_conv, const int nr, const int n_t, __local float * xt) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int c0 = get_group_id(0) * 16;
    const int t0 = get_group_id(1) * 16;
    const int s  = get_group_id(2);
    const int w  = 16 + d_conv - 1;
    // load: row ly of the tile (channel c0 + ly), elements lx, lx + 16, ...
    const int cr = min(c0 + ly, nr - 1);
    __global const float * xr = (__global const float *) (sx + offx + (ulong) cr*nbx1 + (ulong) s*nbx2) + t0;
    const int nx = min(w, n_t + d_conv - 1 - t0); // valid inputs of the row from t0
    for (int e = lx; e < w; e += 16) {
        xt[ly*w + e] = e < nx ? xr[e] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int ch = c0 + lx;
    const int t  = t0 + ly;
    if (ch < nr && t < n_t) {
        __global const float * wr = (__global const float *) (c + offc + (ulong) ch*nbc1);
        float sum = 0.0f;
        for (int k = 0; k < d_conv; ++k) {
            sum += xt[lx*w + ly + k] * wr[k];
        }
        ((__global float *) (dst + offd + (ulong) t*nb1 + (ulong) s*nb2))[ch] = sum;
    }
}

#define MMV_NV 4

inline float4 mmv_ld(__global const char * row, const int k, const int ne00, const int f16) {
    if (k + 4 <= ne00) {
        return f16 ? vload_half4(0, (__global const half *) row + k) : vload4(0, (__global const float *) row + k);
    }
    float4 x = (float4) (0.0f);
    if (f16) {
        __global const half * p = (__global const half *) row + k;
        x.x = vload_half(0, p);
        if (k + 1 < ne00) x.y = vload_half(1, p);
        if (k + 2 < ne00) x.z = vload_half(2, p);
    } else {
        __global const float * p = (__global const float *) row + k;
        x.x = p[0];
        if (k + 1 < ne00) x.y = p[1];
        if (k + 2 < ne00) x.z = p[2];
    }
    return x;
}

inline float4 mmv_ldy(__global const float * yv, const int k, const int ne00) {
    if (k + 4 <= ne00) {
        return vload4(0, yv + k);
    }
    float4 y = (float4) (0.0f);
    y.x = yv[k];
    if (k + 1 < ne00) y.y = yv[k + 1];
    if (k + 2 < ne00) y.z = yv[k + 2];
    return y;
}

__kernel void kernel_mul_mat_vec_f(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                   __global char * dst, const ulong offd,
                                   const int ne00, const int ne01, const int ne11,
                                   const ulong nb01, const ulong nb02, const ulong nb03,
                                   const ulong nb11, const ulong nb12, const ulong nb13,
                                   const ulong nb1, const ulong nb2, const ulong nb3,
                                   const int r2, const int r3, const int src0_f16, const int lpr, __local float * red) {
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int l    = lid % lpr;
    const int i01  = get_group_id(0) * (lsz / lpr) + lid / lpr;
    const int i02  = get_group_id(1);
    const int i13  = get_group_id(2);
    const int i03  = i13 / r3;
    const int nvec = r2 * ne11; // vector v: token i11 = v / r2, head i12 = i02*r2 + v % r2
    __global const char * row = src0 + off0 + (ulong) min(i01, ne01 - 1) * nb01 + (ulong) i02 * nb02 + (ulong) i03 * nb03;

    for (int v0 = 0; v0 < nvec; v0 += MMV_NV) {
        __global const float * yv[MMV_NV];
        #pragma unroll
        for (int c = 0; c < MMV_NV; ++c) {
            const int v = min(v0 + c, nvec - 1);
            yv[c] = (__global const float *) (src1 + off1 + (ulong) (v / r2) * nb11 + (ulong) (i02*r2 + v % r2) * nb12 + (ulong) i13 * nb13);
        }
        float acc[MMV_NV] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int k = 4*l; k < ne00; k += 4*lpr) {
            const float4 x = mmv_ld(row, k, ne00, src0_f16);
            #pragma unroll
            for (int c = 0; c < MMV_NV; ++c) {
                const float4 y = mmv_ldy(yv[c], k, ne00);
                acc[c] = fma(x.x, y.x, fma(x.y, y.y, fma(x.z, y.z, fma(x.w, y.w, acc[c]))));
            }
        }
        // reduction of the lpr lanes of every row, for the MMV_NV vectors
        #pragma unroll
        for (int c = 0; c < MMV_NV; ++c) {
            red[c*lsz + lid] = acc[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lpr >> 1; s > 0; s >>= 1) {
            if (l < s) {
                #pragma unroll
                for (int c = 0; c < MMV_NV; ++c) {
                    red[c*lsz + lid] += red[c*lsz + lid + s];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (l < MMV_NV && v0 + l < nvec && i01 < ne01) {
            const int v = v0 + l;
            __global float * d = (__global float *) (dst + offd + (ulong) (v / r2) * nb1 + (ulong) (i02*r2 + v % r2) * nb2 + (ulong) i13 * nb3);
            d[i01] = red[l*lsz + (lid - l)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CLC";

// Matrix x several vectors for short batches (Q4_K/Q5_K/Q6_K weights, NC columns per launch), tuned on
// the GT 720M with examples/bench-gemm (BENCH_GEMM_MMVN=1). The weights are read and dequantized once for
// NC columns instead of once per column:
//   - same lane layout as dmmv_kq_l8_source: 8 lanes per row, every lane handles 32 weights of each
//     256-weight super-block (4 weights of each of its 8 chunks of 32);
//   - every group of 8 lanes handles R consecutive rows, so every y value it loads feeds R FMAs (the
//     loads, not the FMAs, are the limit on Fermi) and y is re-read R times less;
//   - every lane computes the full value of its weights (d*sc*q - dmin*m) once per super-block and
//     multiplies it by the NC columns with fma chains; the scales are decoded per sub-block from the
//     packed words, which keeps fewer registers alive than 16 floats per row;
//   - the columns are first packed by kernel_mmvn_pack into yi: for every chunk of 32 values of k, the
//     NC columns one after another (zeros for the missing columns), so a lane reads every column from
//     one pointer with constant offsets and the 8 lanes of a row read 128 consecutive bytes. vload4 is
//     kept on purpose: float4 pointers (128-bit loads) are 1.6-2.5x slower here;
//   - the 8 partial sums of every row and column are reduced in local memory in one step.
// Compiled once per NC, with NC, R and LWG (work-group size, a multiple of 8) defined before the
// source. Requires nrows % R == 0.
static const char * be_mmvn_source = R"CLC(
// scale and min of sub-block q (0..7) of a Q4_K/Q5_K super-block, from its 3 packed words (get_scale_min_k4)
inline uint mmvn_sc(const uint s0, const uint s2, const int q) {
    return q < 4 ? (s0 >> (8*q)) & 63 : ((s2 >> (8*(q - 4))) & 0xF) | (((s0 >> (8*(q - 4) + 6)) & 3) << 4);
}
inline uint mmvn_mn(const uint s1, const uint s2, const int q) {
    return q < 4 ? (s1 >> (8*q)) & 63 : ((s2 >> (8*(q - 4) + 4)) & 0xF) | (((s1 >> (8*(q - 4) + 6)) & 3) << 4);
}

inline uint mmvn_q6_ld_u32(__global const uint * pa, const int idx, const uint sh, const uint hmask) {
    return (pa[idx] >> sh) | ((pa[idx + 1] << 16) & hmask);
}

// acc[r][c] += dot(wf[r], y of chunk q and column c), for the R rows of the lane group
inline void mmvn_chunk(float acc[R][NC], const float4 * wf, __global const float * yb, const int q) {
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float4 y = vload4(0, yb + (q*NC + c)*32);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r][c] = fma(wf[r].x, y.x, acc[r][c]);
            acc[r][c] = fma(wf[r].y, y.y, acc[r][c]);
            acc[r][c] = fma(wf[r].z, y.z, acc[r][c]);
            acc[r][c] = fma(wf[r].w, y.w, acc[r][c]);
        }
    }
}

inline void mmvn_reduce_store(__local float * tmp, const int lid, const int j, float acc[R][NC],
                              const int row0, const int nrows, const int ny, __global float * dst, const int ldd) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            tmp[(r*NC + c)*(LWG + 1) + lid] = acc[r][c];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int l0 = lid & ~7;
    #pragma unroll
    for (int i0 = 0; i0 < R*NC; i0 += 8) {
        const int idx = i0 + j;
        const int r   = idx / NC;
        const int c   = idx % NC;
        if (idx < R*NC && c < ny && row0 + r < nrows) {
            float s = 0.0f;
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                s += tmp[idx*(LWG + 1) + l0 + l];
            }
            dst[c*ldd + row0 + r] = s;
        }
    }
}

inline void mmvn_zero(float acc[R][NC]) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
}

// y (ny columns of K floats, column stride ldy floats) -> yi (K*NC floats)
__kernel void kernel_mmvn_pack(__global const char * y_base, const ulong yoff, const int ldy, const int ny,
                               const int K, __global float * yi) {
    const int o = get_global_id(0);
    if (o >= K*NC) {
        return;
    }
    const int e = o & 31;
    const int c = (o >> 5) % NC;
    const int q = (o >> 5) / NC;
    __global const float * y = (__global const float *) (y_base + yoff);
    yi[o] = c < ny ? y[c*ldy + q*32 + e] : 0.0f;
}

// block_q4_K (144 bytes): half d, half dmin, uchar scales[12], uchar qs[128]; part p (64 weights) of
// lane j: quant word 8p + j, low nibbles = chunk 2p, high nibbles = chunk 2p + 1
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void kernel_mmvn_q4_K(__global const uchar * xq_base, const ulong xoff, __global const float * yi,
                      __global char * dst_base, const ulong doff, const int ldd,
                      const int ncols, const int nrows, const int ny) {
    __local float tmp[NC*R*(LWG + 1)];
    __global const uchar * xq  = xq_base + xoff;
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row0 = (get_group_id(0) * (LWG/8) + (lid >> 3)) * R;
    const int nb   = ncols / 256;

    float acc[R][NC];
    mmvn_zero(acc);
    if (row0 < nrows) {
        const int rs = nb * 144;
        __global const uchar * blk = xq + (size_t) row0 * rs;
        __global const float * yb  = yi + 4*j;
        for (int i = 0; i < nb; ++i, blk += 144, yb += 256*NC) {
            float d[R];
            float dmin[R];
            uint  s0[R];
            uint  s1[R];
            uint  s2[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float2 dm = vload_half2(0, (__global const half *) (blk + r*rs));
                __global const uint * sp = (__global const uint *) (blk + r*rs + 4);
                d[r]    = dm.x;
                dmin[r] = dm.y;
                s0[r]   = sp[0];
                s1[r]   = sp[1];
                s2[r]   = sp[2];
            }
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint w[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    w[r] = ((__global const uint *) (blk + r*rs + 16))[j + 8*p];
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int q = 2*p + h;
                    float4 wf[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        wf[r] = convert_float4(as_uchar4((w[r] >> (4*h)) & 0x0F0F0F0Fu)) * (d[r] * (float) mmvn_sc(s0[r], s2[r], q))
                              - dmin[r] * (float) mmvn_mn(s1[r], s2[r], q);
                    }
                    mmvn_chunk(acc, wf, yb, q);
                }
            }
        }
    }
    mmvn_reduce_store(tmp, lid, j, acc, row0, nrows, ny, dst, ldd);
}

// block_q5_K (176 bytes): half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128]; qh byte l
// (l = 4j..4j+3) holds the 5th bit of weight l of every sub-block (bit q -> sub-block q)
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void kernel_mmvn_q5_K(__global const uchar * xq_base, const ulong xoff, __global const float * yi,
                      __global char * dst_base, const ulong doff, const int ldd,
                      const int ncols, const int nrows, const int ny) {
    __local float tmp[NC*R*(LWG + 1)];
    __global const uchar * xq  = xq_base + xoff;
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row0 = (get_group_id(0) * (LWG/8) + (lid >> 3)) * R;
    const int nb   = ncols / 256;

    float acc[R][NC];
    mmvn_zero(acc);
    if (row0 < nrows) {
        const int rs = nb * 176;
        __global const uchar * blk = xq + (size_t) row0 * rs;
        __global const float * yb  = yi + 4*j;
        for (int i = 0; i < nb; ++i, blk += 176, yb += 256*NC) {
            float d[R];
            float dmin[R];
            uint  s0[R];
            uint  s1[R];
            uint  s2[R];
            uint  hq[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float2 dm = vload_half2(0, (__global const half *) (blk + r*rs));
                __global const uint * sp = (__global const uint *) (blk + r*rs + 4);
                d[r]    = dm.x;
                dmin[r] = dm.y;
                s0[r]   = sp[0];
                s1[r]   = sp[1];
                s2[r]   = sp[2];
                hq[r]   = ((__global const uint *) (blk + r*rs + 16))[j];
            }
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint w[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    w[r] = ((__global const uint *) (blk + r*rs + 48))[j + 8*p];
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int q = 2*p + h;
                    float4 wf[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint qv = ((w[r] >> (4*h)) & 0x0F0F0F0Fu) | (((hq[r] >> q) & 0x01010101u) << 4);
                        wf[r] = convert_float4(as_uchar4(qv)) * (d[r] * (float) mmvn_sc(s0[r], s2[r], q))
                              - dmin[r] * (float) mmvn_mn(s1[r], s2[r], q);
                    }
                    mmvn_chunk(acc, wf, yb, q);
                }
            }
        }
    }
    mmvn_reduce_store(tmp, lid, j, acc, row0, nrows, ny, dst, ldd);
}

// block_q6_K (210 bytes): uchar ql[128], uchar qh[64], char scales[16], half d; only 2-byte aligned
// (see dequantize_mul_mat_vec_q6_K_l8: xoff must be a multiple of 4, and the alignment of every
// super-block follows from the parity of its index). Lane j handles l = 4j..4j+3 of both halves h and
// both selectors sel: low nibbles = chunk 4h + sel, high nibbles = chunk 4h + sel + 2.
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void kernel_mmvn_q6_K(__global const uchar * xq_base, const ulong xoff, __global const float * yi,
                      __global char * dst_base, const ulong doff, const int ldd,
                      const int ncols, const int nrows, const int ny) {
    __local float tmp[NC*R*(LWG + 1)];
    __global const uchar * xq  = xq_base + xoff;
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row0 = (get_group_id(0) * (LWG/8) + (lid >> 3)) * R;
    const int nb   = ncols / 256;
    const int l    = 4*j;
    const int is   = j >> 2;

    float acc[R][NC];
    mmvn_zero(acc);
    if (row0 < nrows) {
        const int n0 = row0 * nb; // index of the first super-block of the first row
        __global const uchar * blk = xq + (size_t) n0 * 210;
        __global const float * yb  = yi + 4*j;
        for (int i = 0; i < nb; ++i, blk += 210, yb += 256*NC) {
            __global const uint * pa[R];
            __global const char * sc[R];
            uint  sh[R];
            uint  hmask[R];
            float d[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint odd = (uint) ((n0 + r*nb + i) & 1) << 1; // 0 or 2
                __global const uchar * b = blk + r*nb*210;
                pa[r]    = (__global const uint *) (b - odd);
                sc[r]    = (__global const char *) (b + 192);
                sh[r]    = odd << 3;
                hmask[r] = 0u - (odd >> 1);
                d[r]     = vload_half(0, (__global const half *) (b + 208));
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint hq[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    hq[r] = mmvn_q6_ld_u32(pa[r], (128 + 32*h + l)/4, sh[r], hmask[r]);
                }
                #pragma unroll
                for (int sel = 0; sel < 2; ++sel) {
                    uint  w[R];
                    float dlo[R];
                    float dhi[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        w[r]   = mmvn_q6_ld_u32(pa[r], (64*h + 32*sel + l)/4, sh[r], hmask[r]);
                        dlo[r] = d[r] * (float) sc[r][8*h + 2*sel + is];
                        dhi[r] = d[r] * (float) sc[r][8*h + 2*sel + 4 + is];
                    }
                    float4 wf[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint qlo = (w[r] & 0x0F0F0F0Fu) | (((hq[r] >> (2*sel)) & 0x03030303u) << 4);
                        wf[r] = convert_float4(as_uchar4(qlo)) * dlo[r] - 32.0f*dlo[r];
                    }
                    mmvn_chunk(acc, wf, yb, 4*h + sel);
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint qhi = ((w[r] >> 4) & 0x0F0F0F0Fu) | (((hq[r] >> (4 + 2*sel)) & 0x03030303u) << 4);
                        wf[r] = convert_float4(as_uchar4(qhi)) * dhi[r] - 32.0f*dhi[r];
                    }
                    mmvn_chunk(acc, wf, yb, 4*h + sel + 2);
                }
            }
        }
    }
    mmvn_reduce_store(tmp, lid, j, acc, row0, nrows, ny, dst, ldd);
}

// the same with Q6_K weights in the q6r layout (see be_q6r_common, compiled before this source): lane j
// handles half h = j >> 2 and positions 8*(j & 3) .. + 7 of it, i.e. elements 8*(j & 3) .. + 7 of chunks
// 4h + g (g = 0..3) of every super-block
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void kernel_mmvn_q6r(__global const uchar * xq_base, const ulong xoff, __global const float * yi,
                     __global char * dst_base, const ulong doff, const int ldd,
                     const int ncols, const int nrows, const int ny) {
    __local float tmp[NC*R*(LWG + 1)];
    __global const uchar * xq  = xq_base + xoff;
    __global       float * dst = (__global float *) (dst_base + doff);
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int h    = j >> 2;
    const int p    = j & 3;
    const int row0 = (get_group_id(0) * (LWG/8) + (lid >> 3)) * R;
    const int nb   = ncols / 256;

    float acc[R][NC];
    mmvn_zero(acc);
    if (row0 < nrows) {
        const size_t rs = (size_t) nb * 210;
        __global const uchar * rb = xq + (size_t) row0 * rs;
        __global const float * yb = yi + 8*p;
        for (int i = 0; i < nb; ++i, yb += 256*NC) {
            uint2 a[R];
            uint2 b[R];
            uint2 c[R];
            float ds[R][4];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                __global const uchar * rr = rb + r*rs;
                a[r] = vload2(0, (__global const uint *) (rr + (size_t) i*128 + 8*j));
                b[r] = vload2(0, (__global const uint *) (rr + (size_t) i*128 + 64 + 8*j));
                c[r] = vload2(0, (__global const uint *) (rr + (size_t) nb*128 + (size_t) i*64 + 8*j));
                const char4 s = as_char4(((__global const uint *) (rr + (size_t) nb*192 + (size_t) i*16))[2*h + (p >> 1)]);
                const float d = vload_half(i, (__global const half *) (rr + (size_t) nb*208));
                ds[r][0] = d * (float) s.x;
                ds[r][1] = d * (float) s.y;
                ds[r][2] = d * (float) s.z;
                ds[r][3] = d * (float) s.w;
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float4 wf[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint wa = e ? a[r].y : a[r].x;
                        const uint wb = e ? b[r].y : b[r].x;
                        const uint wc = e ? c[r].y : c[r].x;
                        const uint q  = g == 0 ? Q6R_G0(wa, wb, wc) : g == 1 ? Q6R_G1(wa, wb, wc) :
                                        g == 2 ? Q6R_G2(wa, wb, wc) : Q6R_G3(wa, wb, wc);
                        wf[r] = convert_float4(as_uchar4(q)) * ds[r][g] - 32.0f*ds[r][g];
                    }
                    mmvn_chunk(acc, wf, yb + 4*e, 4*h + g);
                }
            }
        }
    }
    mmvn_reduce_store(tmp, lid, j, acc, row0, nrows, ny, dst, ldd);
}
)CLC";

// Matrix x NC vectors (NC = 2..4: the verification batches of multi-token prediction and prompts of 2-4
// tokens) with the tuned matrix x vector kernels of be_q45_source (Q4_K/Q5_K) and be_q6r_source (q6r):
// the same lanes, the same decoding of every super-block, done once for the NC columns, and the NC
// columns of y staged in local memory (column c at yl + c*(stride of a staged column)). Every column gets
// exactly the operations of the one-column kernel in the same order (the same fma chains, the same sums
// of y, the same partial sums per chunk of lch values for Q4_K/Q5_K, the same reduction tree), so that
// decoding one token or verifying it with drafts gives the same logits. That holds bit for bit when the
// compiler keeps that order, as the one of the NVIDIA 390 driver does on the GT 720M (checked with whole
// batches); the kernels are built with -cl-fast-relaxed-math, which lets a compiler reorder floating point
// operations, and PoCL gives differences of about 1e-4 (as the mmvn kernels on the GT 720M). y is staged in chunks of sch values (a divisor of lch), so that NC
// columns still fit the local memory. Compiled once per NC, with NC defined before the source (and
// be_q6r_common).
// helpers of the kernels for NC columns (be_dmmvn_source, be_q8_source), compiled before them with NC defined
static const char * be_ncol_common = R"CLC(
inline float dmmvn_dot(const float4 f, const float4 y) {
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

// sum of the 8 partial sums of every row in the order of the reduction tree of the one-column kernels
// (strides 4, 2, 1), one result per row and column
inline void dmmvn_reduce_store(__local float * tmp, const int lid, const int lsz, const int j, const float acc[NC],
                               const int row, const int nrows, __global float * dst, const int ldd) {
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        tmp[c*(lsz + 1) + lid] = acc[c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (j < NC && row < nrows) {
        __local const float * t = tmp + j*(lsz + 1) + (lid & ~7);
        dst[(size_t) j*ldd + row] = ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
)CLC";

static const char * be_dmmvn_source = R"CLC(
// ---- Q4_K / Q5_K (see be_q45_source) ----

// y[k0 .. k0 + len) of the NC columns (column stride ldy floats) -> yl + c*sch, sums of every 8 values ->
// ysl + c*(sch/8)
inline void q45n_stage(__global const float * y, const int ldy, const int k0, const int len, const int sch,
                       __local float * yl, __local float * ysl, const int lid, const int lsz) {
    const int n8 = len/8;
    for (int t = lid; t < NC*n8; t += lsz) {
        const int c = t / n8;
        const int e = t - c*n8;
        __global const float * yc = y + (size_t) c*ldy + k0 + 8*e;
        const float4 u = vload4(0, yc);
        const float4 w = vload4(0, yc + 4);
        vstore4(u, 0, yl + c*sch + 8*e);
        vstore4(w, 0, yl + c*sch + 8*e + 4);
        ysl[c*(sch/8) + e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
    }
}

inline void q45n_q(float sd[NC], const float s, const uint q, __local const float * yb, const int sch) {
    const float4 f = convert_float4(as_uchar4(q));
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        sd[c] = fma(s, dmmvn_dot(f, vload4(0, yb + c*sch)), sd[c]);
    }
}

// q45_part for NC columns, accumulating into part
inline void q45n_part(__global const uchar * blk, const int nbc, const int bs, const int j, const int sch,
                      __local const float * yl, __local const float4 * ysl4, float part[NC]) {
    const int jj = j & 3;
    for (int i = 0; i < nbc; ++i, blk += bs) {
        const float2 dm = vload_half2(0, (__global const half *) blk);
        __global const uint * sp = (__global const uint *) (blk + 4);
        const uint s0 = sp[0];
        const uint s1 = sp[1];
        const uint s2 = sp[2];
        const float4 sc_a = convert_float4(as_uchar4(s0 & 0x3F3F3F3Fu));
        const float4 sc_b = convert_float4(as_uchar4((s2 & 0x0F0F0F0Fu) | ((s0 >> 2) & 0x30303030u)));
        const uint   mlo  = (s1 >> (8*jj)) & 63;
        const uint   mhi  = ((s2 >> (8*jj + 4)) & 0xF) | (((s1 >> (8*jj + 6)) & 3) << 4);
        const float  mj   = (float) (j < 4 ? mlo : mhi);
        uint hq = 0;
        __global const uint * qs = (__global const uint *) (blk + 16);
        if (bs == 176) {
            hq = qs[j];
            qs = (__global const uint *) (blk + 48);
        }
        const uint w0 = qs[j];
        const uint w1 = qs[j +  8];
        const uint w2 = qs[j + 16];
        const uint w3 = qs[j + 24];
        __local const float * yb = yl + i*QK_K + 4*j;
        float sd[NC];
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            sd[c] = 0.0f;
        }
        q45n_q(sd, sc_a.x, ( w0       & 0x0F0F0F0Fu) | ((( hq       & 0x01010101u)) << 4), yb      , sch);
        q45n_q(sd, sc_a.y, ((w0 >> 4) & 0x0F0F0F0Fu) | ((((hq >> 1) & 0x01010101u)) << 4), yb +  32, sch);
        q45n_q(sd, sc_a.z, ( w1       & 0x0F0F0F0Fu) | ((((hq >> 2) & 0x01010101u)) << 4), yb +  64, sch);
        q45n_q(sd, sc_a.w, ((w1 >> 4) & 0x0F0F0F0Fu) | ((((hq >> 3) & 0x01010101u)) << 4), yb +  96, sch);
        q45n_q(sd, sc_b.x, ( w2       & 0x0F0F0F0Fu) | ((((hq >> 4) & 0x01010101u)) << 4), yb + 128, sch);
        q45n_q(sd, sc_b.y, ((w2 >> 4) & 0x0F0F0F0Fu) | ((((hq >> 5) & 0x01010101u)) << 4), yb + 160, sch);
        q45n_q(sd, sc_b.z, ( w3       & 0x0F0F0F0Fu) | ((((hq >> 6) & 0x01010101u)) << 4), yb + 192, sch);
        q45n_q(sd, sc_b.w, ((w3 >> 4) & 0x0F0F0F0Fu) | ((((hq >> 7) & 0x01010101u)) << 4), yb + 224, sch);
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 ys4 = ysl4[c*(sch/32) + 8*i + j];
            const float  ysj = (ys4.x + ys4.y) + (ys4.z + ys4.w);
            part[c] = fma(dm.x, sd[c], fma(-dm.y * mj, ysj, part[c]));
        }
    }
}

// lch: chunk of the one-column kernel (its partial sums are added at the end of every chunk of lch
// values); sch: chunk staged in local memory (a divisor of lch). With ncols <= sch, y is staged once and
// every work-group handles T groups of rows (as dmmv_q45_one).
// Local memory: tmp (NC*(local size + 1) floats), yl4 (NC*sch floats), ysl4 (NC*sch/8 floats).
inline void dmmvn_q45(__global const uchar * xq, __global const float * y, const int ldy, __global float * dst, const int ldd,
                      const int ncols, const int nrows, const int lch, const int sch, const int T, const int bs,
                      __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int nb  = ncols / QK_K;
    const int one = ncols <= sch;
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    if (one) {
        q45n_stage(y, ldy, 0, ncols, sch, yl, ysl, lid, lsz);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (lsz >> 3) + (lid >> 3);
        __global const uchar * blk = xq + (size_t) min(row, nrows - 1) * nb * bs;
        float acc[NC];
        float part[NC];
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c]  = 0.0f;
            part[c] = 0.0f;
        }
        for (int k0 = 0; k0 < ncols; k0 += sch) {
            const int len = min(sch, ncols - k0);
            if (!one) {
                if (k0 > 0) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                q45n_stage(y, ldy, k0, len, sch, yl, ysl, lid, lsz);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            q45n_part(blk + (size_t) (k0 / QK_K) * bs, len / QK_K, bs, j, sch, yl, ysl4, part);
            if ((k0 + len) % lch == 0 || k0 + len == ncols) {
                #pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c] += part[c];
                    part[c] = 0.0f;
                }
            }
        }
        dmmvn_reduce_store(tmp, lid, lsz, j, acc, row, nrows, dst, ldd);
    }
}

#define DMMVN_Q45_KERNEL(name, bs) \
__kernel void kernel_dmmvn_##name(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, const int ldy, \
                                  __global char * dst_base, const ulong doff, const int ldd, const int ncols, const int nrows, \
                                  const int lch, const int sch, const int T, \
                                  __local float * tmp, __local float4 * yl4, __local float4 * ysl4) { \
    dmmvn_q45(xq_base + xoff, (__global const float *) (y_base + yoff), ldy, (__global float *) (dst_base + doff), ldd, \
              ncols, nrows, lch, sch, T, bs, tmp, yl4, ysl4); \
}

DMMVN_Q45_KERNEL(q4_K, 144)
DMMVN_Q45_KERNEL(q5_K, 176)

// ---- q6r (see be_q6r_source) ----

#define Q6R_YOFF(c) ((c) + (((c) >> 7) << 2))
#define Q6R_LDY(p, o) (((__local const float4 *) ((p) + (o)))[0])

// group gi of one super-block (quants q0, q1, scale sv) for the NC columns: Q6R_GRP of be_q6r_source.
// yo: offset of the lane in a staged column, ysi: float4 of its sums of y in a column
inline void q6rn_grp(float sum[NC], const uint q0, const uint q1, const float sv, const int gi, const int o,
                     __local const float * yl, const int ycs, const int yo, __local const float4 * ysl4, const int ysi,
                     const int sch) {
    const float4 f0 = convert_float4(as_uchar4(q0));
    const float4 f1 = convert_float4(as_uchar4(q1));
    #pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        __local const float * yi = yl + cc*ycs + yo;
        const float yg = ((__local const float *) (ysl4 + cc*(sch/32) + ysi))[gi];
        sum[cc] = fma(sv, fma(-32.0f, yg, dmmvn_dot(f0, Q6R_LDY(yi, o)) + dmmvn_dot(f1, Q6R_LDY(yi, o + 4))), sum[cc]);
    }
}

// dmmv_q6r for NC columns. Local memory: tmp (NC*(local size + 1) floats), yl4 (NC*(sch + sch/32)
// floats), ysl4 (NC*sch/8 floats)
inline void dmmvn_q6r(__global const uchar * xq, __global const float * y, const int ldy, __global float * dst, const int ldd,
                      const int ncols, const int nrows, const int sch,
                      __local float * tmp, __local float4 * yl4, __local float4 * ysl4, const int v2) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int h   = j >> 2;
    const int p   = j & 3;
    const int nb  = ncols / QK_K;
    const int row = get_group_id(0) * (lsz >> 3) + (lid >> 3);
    const int ycs = sch + sch/32; // floats of a staged column
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;

    __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 210;
    __global const uchar * pq = rb + 8*j;
    __global const uchar * ph = rb + (size_t) nb*128 + 8*j;
    __global const uint  * ps = (__global const uint *) (rb + (size_t) nb*192) + 2*h + (p >> 1);
    __global const half  * pd = (__global const half *) (rb + (size_t) nb*208);

    float acc[NC];
    #pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        acc[cc] = 0.0f;
    }
    for (int k0 = 0; k0 < ncols; k0 += sch) {
        const int len = min(sch, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const int n8 = len/8;
        for (int t = lid; t < NC*n8; t += lsz) {
            const int cc = t / n8;
            const int e  = t - cc*n8;
            __global const float * yc = y + (size_t) cc*ldy + k0 + 8*e;
            const float4 u = vload4(0, yc);
            const float4 w = vload4(0, yc + 4);
            vstore4(u, 0, yl + cc*ycs + Q6R_YOFF(8*e));
            vstore4(w, 0, yl + cc*ycs + Q6R_YOFF(8*e) + 4);
            ysl[cc*(sch/8) + ((e & ~15) | ((e & 3) << 2) | ((e >> 2) & 3))] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nbc = len / QK_K;
        for (int i = 0; i < nbc; ++i) {
            uint2 a;
            uint2 b;
            uint2 c;
            if (v2) {
                a = ((__global const uint2 *) pq)[0];
                b = ((__global const uint2 *) (pq + 64))[0];
                c = ((__global const uint2 *) ph)[0];
            } else {
                a = vload2(0, (__global const uint *) pq);
                b = vload2(0, (__global const uint *) (pq + 64));
                c = vload2(0, (__global const uint *) ph);
            }
            const char4 s  = as_char4(*ps);
            const float d  = vload_half(0, pd);
            const int   yo  = Q6R_YOFF(i*QK_K + 128*h + 8*p);
            const int   ysi = 8*i + 4*h + p;
            float sum[NC];
            #pragma unroll
            for (int cc = 0; cc < NC; ++cc) {
                sum[cc] = 0.0f;
            }
            q6rn_grp(sum, Q6R_G0(a.x, b.x, c.x), Q6R_G0(a.y, b.y, c.y), (float) s.x, 0,  0, yl, ycs, yo, ysl4, ysi, sch);
            q6rn_grp(sum, Q6R_G1(a.x, b.x, c.x), Q6R_G1(a.y, b.y, c.y), (float) s.y, 1, 32, yl, ycs, yo, ysl4, ysi, sch);
            q6rn_grp(sum, Q6R_G2(a.x, b.x, c.x), Q6R_G2(a.y, b.y, c.y), (float) s.z, 2, 64, yl, ycs, yo, ysl4, ysi, sch);
            q6rn_grp(sum, Q6R_G3(a.x, b.x, c.x), Q6R_G3(a.y, b.y, c.y), (float) s.w, 3, 96, yl, ycs, yo, ysl4, ysi, sch);
            #pragma unroll
            for (int cc = 0; cc < NC; ++cc) {
                acc[cc] = fma(d, sum[cc], acc[cc]);
            }
            pq += 128;
            ph += 64;
            ps += 4;
            pd += 1;
        }
    }
    dmmvn_reduce_store(tmp, lid, lsz, j, acc, row, nrows, dst, ldd);
}

__kernel void kernel_dmmvn_q6r(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, const int ldy,
                               __global char * dst_base, const ulong doff, const int ldd, const int ncols, const int nrows, const int sch,
                               __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    dmmvn_q6r(xq_base + xoff, (__global const float *) (y_base + yoff), ldy, (__global float *) (dst_base + doff), ldd,
              ncols, nrows, sch, tmp, yl4, ysl4, 0);
}

__kernel void kernel_dmmvn_q6r_v2(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, const int ldy,
                                  __global char * dst_base, const ulong doff, const int ldd, const int ncols, const int nrows, const int sch,
                                  __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    dmmvn_q6r(xq_base + xoff, (__global const float *) (y_base + yoff), ldy, (__global float *) (dst_base + doff), ldd,
              ncols, nrows, sch, tmp, yl4, ysl4, 1);
}
)CLC";

// Q8_0 weights in the "q8r" layout of the OpenCL buffers (see ggml_cl_q8r): block_q8_0 (half d, int8 qs[32]:
// 34 bytes, only 2-byte aligned) is reordered inside every row of nb blocks as qs[nb][32] followed by d[nb],
// so the row keeps its size (34*nb bytes) and every row is still found at row*nb1. Rows with an even number
// of blocks (ne[0] % 64 == 0) are 4-byte aligned, and 16-byte aligned with ne[0] % 256 == 0 (v2 kernels).
// Matrix x NC vectors (NC = 1..4; compiled once per NC, with be_ncol_common): 8 lanes per row; in every step
// of 128 weights lane j takes the 16 weights 16j..16j+15 (one load of 16 bytes, half a block, and its d),
// converts them once and adds them to the NC columns staged in local memory; every column gets the same
// operations in the same order for any NC, so the matrix x vector kernel (NC = 1) and the verification
// batches of multi-token prediction (NC = 2..4) give the same results (bit for bit with the compiler of the
// GT 720M; see be_dmmvn_source). y in chunks of sch values
// (a multiple of 128); with ncols <= sch, y is staged once and every work-group handles T groups of rows.
// GT 720M (Qwen3.5-0.8B Q8_0): 8.2 GB/s with one column (80% of the ceiling; 35-40 registers); 2 columns cost
// 1.59 times one, limited by instructions (the second column adds its FMAs and local loads; loading the 16
// bytes in two halves took 42 registers instead of 44-48 and changed nothing), so the drafts of MTP rarely
// pay off with Q8_0 weights there.
static const char * be_q8_source = R"CLC(
inline void q8n_stage(__global const float * y, const int ldy, const int k0, const int len, const int sch,
                      __local float * yl, const int lid, const int lsz) {
    const int n4 = len/4;
    for (int t = lid; t < NC*n4; t += lsz) {
        const int c = t / n4;
        const int e = t - c*n4;
        vstore4(vload4(0, y + (size_t) c*ldy + k0 + 4*e), 0, yl + c*sch + 4*e);
    }
}

// 4 weights (one 32-bit word of quants) x the 4 values of y at yb of every column
inline void q8n_grp(float s[NC], const uint q, __local const float * yb, const int sch) {
    const float4 f = convert_float4(as_char4(q));
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float4 y = ((__local const float4 *) (yb + c*sch))[0];
        s[c] = fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, fma(f.x, y.x, s[c]))));
    }
}

// Local memory: tmp (NC*(local size + 1) floats), yl4 (NC*sch floats)
inline void dmmv_q8r(__global const uchar * xq, __global const float * y, const int ldy, __global float * dst, const int ldd,
                     const int ncols, const int nrows, const int sch, const int T,
                     __local float * tmp, __local float4 * yl4, const int v2) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int nb  = ncols / 32;
    const int one = ncols <= sch;
    __local float * yl = (__local float *) yl4;
    if (one) {
        q8n_stage(y, ldy, 0, ncols, sch, yl, lid, lsz);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (lsz >> 3) + (lid >> 3);
        __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 34;
        __global const half  * pd = (__global const half *) (rb + (size_t) nb * 32);
        float acc[NC];
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] = 0.0f;
        }
        for (int k0 = 0; k0 < ncols; k0 += sch) {
            const int len = min(sch, ncols - k0);
            if (!one) {
                if (k0 > 0) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                q8n_stage(y, ldy, k0, len, sch, yl, lid, lsz);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            for (int kk = 16*j; kk < len; kk += 128) {
                const int k = k0 + kk;
                const float d = vload_half(k >> 5, pd);
                __local const float * yb = yl + kk;
                float s[NC];
                #pragma unroll
                for (int c = 0; c < NC; ++c) {
                    s[c] = 0.0f;
                }
                uint4 q;
                if (v2) {
                    q = ((__global const uint4 *) (rb + k))[0];
                } else {
                    q = vload4(0, (__global const uint *) (rb + k));
                }
                q8n_grp(s, q.x, yb,      sch);
                q8n_grp(s, q.y, yb +  4, sch);
                q8n_grp(s, q.z, yb +  8, sch);
                q8n_grp(s, q.w, yb + 12, sch);
                #pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c] = fma(d, s[c], acc[c]);
                }
            }
        }
        dmmvn_reduce_store(tmp, lid, lsz, j, acc, row, nrows, dst, ldd);
    }
}

__kernel void kernel_dmmv_q8r(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, const int ldy,
                              __global char * dst_base, const ulong doff, const int ldd, const int ncols, const int nrows,
                              const int sch, const int T, __local float * tmp, __local float4 * yl4) {
    dmmv_q8r(xq_base + xoff, (__global const float *) (y_base + yoff), ldy, (__global float *) (dst_base + doff), ldd,
             ncols, nrows, sch, T, tmp, yl4, 0);
}

__kernel void kernel_dmmv_q8r_v2(__global const uchar * xq_base, const ulong xoff, __global const char * y_base, const ulong yoff, const int ldy,
                                 __global char * dst_base, const ulong doff, const int ldd, const int ncols, const int nrows,
                                 const int sch, const int T, __local float * tmp, __local float4 * yl4) {
    dmmv_q8r(xq_base + xoff, (__global const float *) (y_base + yoff), ldy, (__global float *) (dst_base + doff), ldd,
             ncols, nrows, sch, T, tmp, yl4, 1);
}

// 4 weights of a q8r row (word e of its quants) -> F32
inline float4 q8r_dq4(__global const uchar * rb, const int nb, const int e) {
    const char4 q = as_char4(((__global const uint *) rb)[e]);
    return vload_half(e >> 3, (__global const half *) (rb + (size_t) nb * 32)) * convert_float4(q);
}

// GET_ROWS with q8r rows (token embeddings): one work-group per row of dst, 4 weights per work-item
__kernel void kernel_get_rows_q8r(__global const char * src0, const ulong off0, __global const char * src1, const ulong off1,
                                  __global char * dst, const ulong offd,
                                  const int nb, const ulong nb01, const ulong nb02, const ulong nb03,
                                  const ulong nb10, const ulong nb11, const ulong nb12,
                                  const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *((__global const int *) (src1 + off1 + i10*nb10 + i11*nb11 + i12*nb12));
    __global const uchar * rb = (__global const uchar *) (src0 + off0 + (ulong) r*nb01 + i11*nb02 + i12*nb03);
    __global       float * d  = (__global float *) (dst + offd + i10*nb1 + i11*nb2 + i12*nb3);
    for (int e = get_local_id(0); e < nb*8; e += get_local_size(0)) {
        vstore4(q8r_dq4(rb, nb, e), 0, d + 4*e);
    }
}

// dequantization of a whole q8r matrix (nrows rows of nb blocks) to F32, for the GEMM of batches
__kernel void kernel_dequant_q8r(__global const uchar * xq_base, const ulong xoff, __global char * dst_base, const ulong doff,
                                 const uint n4, const int nb) {
    const uint gid = get_global_id(0);
    if (gid >= n4) {
        return;
    }
    const uint r = gid / (uint) (nb*8);
    const int  e = (int) (gid - r * (uint) (nb*8));
    vstore4(q8r_dq4(xq_base + xoff + (size_t) r * nb * 34, nb, e), 0, (__global float *) (dst_base + doff) + (size_t) gid * 4);
}
)CLC";

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

std::array<std::string, 5> dequant_str_keys = {
    "KERNEL_NAME", "X_TYPE", "QUANT_K", "QUANT_R", "DEQUANT_FUNC"
};

std::array<std::string, 30> dequant_str_values = {
    "dequantize_row_q4_0", "struct block_q4_0", "QK4_0", "QR4_0", "dequantize_q4_0",
    "dequantize_row_q4_1", "struct block_q4_1", "QK4_1", "QR4_1", "dequantize_q4_1",
    "dequantize_row_q5_0", "struct block_q5_0", "QK5_0", "QR5_0", "dequantize_q5_0",
    "dequantize_row_q5_1", "struct block_q5_1", "QK5_1", "QR5_1", "dequantize_q5_1",
    "dequantize_row_q8_0", "struct block_q8_0", "QK8_0", "QR8_0", "dequantize_q8_0",
    "convert_row_f16", "half", "1", "1", "convert_f16"
};

std::array<std::string, 30> dequant_mul_mat_vec_str_values = {
    "dequantize_mul_mat_vec_q4_0", "struct block_q4_0", "QK4_0", "QR4_0", "dequantize_q4_0",
    "dequantize_mul_mat_vec_q4_1", "struct block_q4_1", "QK4_1", "QR4_1", "dequantize_q4_1",
    "dequantize_mul_mat_vec_q5_0", "struct block_q5_0", "QK5_0", "QR5_0", "dequantize_q5_0",
    "dequantize_mul_mat_vec_q5_1", "struct block_q5_1", "QK5_1", "QR5_1", "dequantize_q5_1",
    "dequantize_mul_mat_vec_q8_0", "struct block_q8_0", "QK8_0", "QR8_0", "dequantize_q8_0",
    "convert_mul_mat_vec_f16", "half", "1", "1", "convert_f16"
};

std::array<std::string, 2> mul_str_keys = {
    "KERNEL_NAME", "TYPE"
};
std::array<std::string, 2> mul_str_values = {
    "mul_f32", "float"
};

static std::string& replace(std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
         s.replace(pos, from.length(), to);
         pos += to.length();
    }
    return s;
}

// Q8_0 rows of the K/V views of flash attention (the cache with -ctk/-ctv q8_0), for kernel_flash_attn_f16
// (be_fattn_source) and kernel_dequant_q8_0_kv. A row of a view starts at byte a in the coordinates of the standard
// layout, where block b of the row is its scale (half) and its 32 quants (34 bytes). In q8r (b0 >= 0, see
// be_q8_source) the row of the view is the part of a row of its base tensor (nq values) that starts at block
// b0 + h*bh of it (h: dimension 2 of the view, the head); a base row holds the quants of its blocks, then their
// scales. ggml_cl_q8_kv_layout gives b0, bh and nq.
static const char * be_q8kv_common = R"CLC(
typedef struct {
    long qa; // quant e of the row at a + qa + qs*(e/32) + e%32
    int  qs;
    long sa; // scale of block b at a + sa + ss*b
    int  ss;
} q8kv_t;

inline q8kv_t q8kv_layout(const int b0, const int bh, const int nq, const int h) {
    q8kv_t l;
    if (b0 >= 0) {
        const long hb0 = b0 + h*bh;
        l.qa = -2*hb0;
        l.qs = 32;
        l.sa = nq - 32*hb0;
        l.ss = 2;
    } else {
        l.qa = 2;
        l.qs = 34;
        l.sa = 0;
        l.ss = 34;
    }
    return l;
}

// value e of the row that starts at a
inline float q8kv_get(__global const char * a, const q8kv_t l, const int e) {
    return vload_half(0, (__global const half *) (a + l.sa + l.ss*(e/32))) * (float) a[l.qa + l.qs*(e/32) + e%32];
}

// values e, e + 1, e + 2, e + 3 (e % 4 == 0) of the row that starts at a
inline float4 q8kv_get4(__global const char * a, const q8kv_t l, const int e) {
    return convert_float4(vload4(0, a + l.qa + l.qs*(e/32) + e%32)) * vload_half(0, (__global const half *) (a + l.sa + l.ss*(e/32)));
}
)CLC";

// Kernels for the operations of the current ggml graphs (SET_ROWS, SWIGLU and the RoPE of ggml_rope_multi).
// Same conventions as backend_ops_source: every tensor is a buffer plus a byte offset and every stride is in bytes.
static const char * be_modern_source = R"CLC(
// SET_ROWS: row r = idx[i01, i02 % ne11, i03 % ne12] of plane (i02, i03) of dst = row (i01, i02, i03) of src0
// (F32), converted to the type of dst (F32 or F16). The indices are I64 or I32. One work-group per row of src0.
__kernel void kernel_set_rows(__global const char * src0, const ulong off0,
                              __global const char * src1, const ulong off1,
                              __global char * dst, const ulong offd,
                              const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                              const int ne11, const int ne12,
                              const ulong nb10, const ulong nb11, const ulong nb12,
                              const ulong nb1, const ulong nb2, const ulong nb3,
                              const int idx_i64, const int dst_f16,
                              const int ne01, const int ne02, const ulong n) {
    // one work-item per value (i00, i01, i02, i03), i00 fastest: rows of any length use the whole device (the
    // rollback of the convolution states of Qwen3.5 writes rows of 3 values)
    const ulong gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    const ulong t0  = gid / (ulong) ne00;
    const int   i00 = (int) (gid - t0*(ulong) ne00);
    const ulong t1  = t0 / (ulong) ne01;
    const int   i01 = (int) (t0 - t1*(ulong) ne01);
    const ulong t2  = t1 / (ulong) ne02;
    const int   i02 = (int) (t1 - t2*(ulong) ne02);
    const int   i03 = (int) t2;
    const int   j2  = i02 - (i02 / ne11)*ne11;
    const int   j3  = i03 - (i03 / ne12)*ne12;
    __global const char * ip = src1 + off1 + i01*nb10 + j2*nb11 + j3*nb12;
    const long r = idx_i64 ? *((__global const long *) ip) : (long) *((__global const int *) ip);
    const float v = ((__global const float *) (src0 + off0 + i01*nb01 + i02*nb02 + i03*nb03))[i00];
    __global char * d = dst + offd + (ulong) r*nb1 + i02*nb2 + i03*nb3;
    if (dst_f16) {
        vstore_half(v, i00, (__global half *) d);
    } else {
        ((__global float *) d)[i00] = v;
    }
}

// SET_ROWS into a Q8_0 tensor (the K/V cache with -ctk/-ctv q8_0), as kernel_set_rows with one work-item per block
// of 32 values, quantized as quantize_row_q8_0_ref of the CPU. Standard layout, or q8r (nq > 0: the rows of dst are
// whole rows of nq values of its base tensor, which holds the quants of a row and then its scales).
__kernel void kernel_set_rows_q8_0(__global const char * src0, const ulong off0,
                                   __global const char * src1, const ulong off1,
                                   __global char * dst, const ulong offd,
                                   const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                                   const int ne11, const int ne12,
                                   const ulong nb10, const ulong nb11, const ulong nb12,
                                   const ulong nb1, const ulong nb2, const ulong nb3,
                                   const int idx_i64, const int ne01, const int ne02, const ulong n, const int nq) {
    const ulong gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    const int   nbl = ne00 / 32;
    const ulong t0  = gid / (ulong) nbl;
    const int   ib  = (int) (gid - t0*(ulong) nbl);
    const ulong t1  = t0 / (ulong) ne01;
    const int   i01 = (int) (t0 - t1*(ulong) ne01);
    const ulong t2  = t1 / (ulong) ne02;
    const int   i02 = (int) (t1 - t2*(ulong) ne02);
    const int   i03 = (int) t2;
    const int   j2  = i02 - (i02 / ne11)*ne11;
    const int   j3  = i03 - (i03 / ne12)*ne12;
    __global const char * ip = src1 + off1 + i01*nb10 + j2*nb11 + j3*nb12;
    const long r = idx_i64 ? *((__global const long *) ip) : (long) *((__global const int *) ip);
    __global const float * x = (__global const float *) (src0 + off0 + i01*nb01 + i02*nb02 + i03*nb03) + ib*32;
    float amax = 0.0f;
    for (int e = 0; e < 32; ++e) {
        amax = max(amax, fabs(x[e]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    __global char * row = dst + offd + (ulong) r*nb1 + i02*nb2 + i03*nb3;
    __global char * qs  = nq > 0 ? row + ib*32 : row + ib*34 + 2;
    vstore_half(d, 0, (__global half *) (nq > 0 ? row + nq + ib*2 : row + ib*34));
    for (int e = 0; e < 32; ++e) {
        qs[e] = (char) round(x[e]*id);
    }
}

// Q8_0 K/V of the decomposed flash attention -> F16 (layouts: be_q8kv_common): value d of row j of head h of a view
// (strides nb1, nb2) to dst[(h*ne1 + j)*ne0 + d], or with tr to dst[(h*ne0 + d)*ne1 + j] (V^T; then dimension 0 of
// the grid runs over j, so that the writes are contiguous)
__kernel void kernel_dequant_q8_0_kv(__global const char * src, const ulong offs, const ulong nb1, const ulong nb2,
                                     const int b0, const int bh, const int nq,
                                     __global char * dst, const ulong offd, const int ne0, const int ne1, const int tr) {
    const int d = tr ? get_global_id(1) : get_global_id(0);
    const int j = tr ? get_global_id(0) : get_global_id(1);
    const int h = get_global_id(2);
    if (d >= ne0 || j >= ne1) {
        return;
    }
    const q8kv_t l = q8kv_layout(b0, bh, nq, h);
    const float  x = q8kv_get(src + offs + (ulong) j*nb1 + (ulong) h*nb2, l, d);
    vstore_half(x, tr ? ((ulong) h*ne0 + d)*ne1 + j : ((ulong) h*ne1 + j)*ne0 + d, (__global half *) (dst + offd));
}

// SWIGLU: dst = silu(a) * b, a and b with their own row strides (two tensors, or the two halves of the rows
// of src0). One work-item per value: dimension 0 of the grid runs along the row, 1 over the rows and 2 over
// the planes (i2 + ne2*i3). silu with exp(-|x|) <= 1, like kernel_silu_f32 (no infinities).
__kernel void kernel_swiglu_f32(__global const char * src0, const ulong off0,
                                __global const char * src1, const ulong off1,
                                __global char * dst, const ulong offd,
                                const int nc, const int ne2,
                                const ulong nb01, const ulong nb02, const ulong nb03,
                                const ulong nb11, const ulong nb12, const ulong nb13,
                                const ulong nb1, const ulong nb2, const ulong nb3) {
    const int i0  = get_global_id(0);
    const int i1  = get_global_id(1);
    const int i23 = get_global_id(2);
    if (i0 >= nc) {
        return;
    }
    const int i2 = i23 % ne2;
    const int i3 = i23 / ne2;
    const float a = ((__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03))[i0];
    const float b = ((__global const float *) (src1 + off1 + i1*nb11 + i2*nb12 + i3*nb13))[i0];
    const float e = exp(-fabs(a));
    const float silu = a >= 0.0f ? a / (1.0f + e) : a * e / (1.0f + e);
    ((__global float *) (dst + offd + i1*nb1 + i2*nb2 + i3*nb3))[i0] = silu * b;
}

// YaRN, same as rope_yarn in ggml-cpu
inline void rope_yarn_m(const float theta_extrap, const float freq_scale, const float corr0, const float corr1,
                        const int i0, const float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = (float) (i0 / 2 - corr0) / max(0.001f, corr1 - corr0);
        const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// ggml_rope_ext and ggml_rope_multi (forward). mode: 0 normal, 1 NEOX, 2 MROPE, 3 IMROPE (interleaved
// M-RoPE, Qwen3-VL and Qwen3.5); MROPE and IMROPE rotate like NEOX, with the position of the section of
// every pair (t, h, w or e: pos holds 4*ne2 positions) and theta_scale^k for every pair k, as in ggml-cpu.
// The n_dims rotated values start at n_offs; the others are copied. One work-item per pair of values of a
// row: pair k < n_dims/2 is rotated, and values 2k and 2k+1 are copied if they are outside the rotated range.
// src0 [ne0, ne1, ne2, ne3] F32 with dimension 0 contiguous; optional F32 frequency factors [n_dims/2].
__kernel void kernel_rope_f32(__global const char * src0, const ulong off0, __global const char * pos, const ulong offp,
                              __global const char * ff, const ulong offff, __global char * dst, const ulong offd,
                              const int ne0, const int ne1, const int ne2,
                              const ulong nb01, const ulong nb02, const ulong nb03,
                              const ulong nb1, const ulong nb2, const ulong nb3,
                              const int n_dims, const int mode, const int has_ff, const int n_offs,
                              const int s0, const int s1, const int s2, const int s3,
                              const float theta_scale, const float freq_scale, const float ext_factor, const float attn_factor,
                              const float corr0, const float corr1) {
    const int k   = get_global_id(0); // pair index
    const int i1  = get_global_id(1);
    const int i23 = get_global_id(2);
    if (k >= ne0/2 || i1 >= ne1) {
        return;
    }
    const int i2 = i23 % ne2;
    const int i3 = i23 / ne2;
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    __global const int   * p = (__global const int *) (pos + offp);

    if (k < n_dims/2) {
        int sel = 0; // position of the section: 0 t, 1 h, 2 w, 3 e
        if (mode >= 2) {
            const int sect_dims = s0 + s1 + s2 + s3;
            const int sector    = k % sect_dims;
            if (mode == 3) {
                if (sector % 3 == 1 && sector < 3*s1) {
                    sel = 1;
                } else if (sector % 3 == 2 && sector < 3*s2) {
                    sel = 2;
                } else if (sector % 3 == 0 && sector < 3*s0) {
                    sel = 0;
                } else {
                    sel = 3;
                }
            } else {
                const int sec_w = s0 + s1;
                if (sector >= s0 && sector < sec_w) {
                    sel = 1;
                } else if (sector >= sec_w && sector < sec_w + s2) {
                    sel = 2;
                } else if (sector >= sec_w + s2) {
                    sel = 3;
                }
            }
        }
        const float pk = (float) p[i2 + sel*ne2];
        const float freq_factor = has_ff ? ((__global const float *) (ff + offff))[k] : 1.0f;
        float c, s;
        rope_yarn_m(pk * pow(theta_scale, (float) k) / freq_factor, freq_scale, corr0, corr1, 2*k, ext_factor, attn_factor, &c, &s);
        const int ia = mode == 0 ? n_offs + 2*k     : n_offs + k;
        const int ib = mode == 0 ? n_offs + 2*k + 1 : n_offs + k + n_dims/2;
        const float x0 = x[ia];
        const float x1 = x[ib];
        d[ia] = x0*c - x1*s;
        d[ib] = x0*s + x1*c;
    }
    const int e = 2*k;
    if (e < n_offs || e >= n_offs + n_dims) {
        d[e]     = x[e];
        d[e + 1] = x[e + 1];
    }
}

// L2_NORM: dst = x / max(||x||, eps), one work-group (power-of-two size) per row; dimension 0 contiguous
__kernel void kernel_l2_norm_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                                 const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                                 const ulong nb1, const ulong nb2, const ulong nb3, const float eps,
                                 __local float * sum) {
    const int i1  = get_group_id(0);
    const int i2  = get_group_id(1);
    const int i3  = get_group_id(2);
    const int lid = get_local_id(0);
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    float s = 0.0f;
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        s += x[i0] * x[i0];
    }
    sum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (lid < k) {
            sum[lid] += sum[lid + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / max(sqrt(sum[0]), eps);
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        d[i0] = x[i0] * scale;
    }
}

// PAD (F32, not circular): dst (contiguous) element (i0, i1, i2, i3) = src0 element (i0 - lp0, i1 - lp1,
// i2 - lp2, i3 - lp3) inside the copied range [lp, ne - rp) of every dimension, 0 outside. One work-item per
// element of dst.
__kernel void kernel_pad_f32(__global const char * src0, const ulong off0, __global char * dst, const ulong offd,
                             const int ne0, const int ne1, const int ne2, const int ne3,
                             const ulong nb00, const ulong nb01, const ulong nb02, const ulong nb03,
                             const int lp0, const int rp0, const int lp1, const int rp1,
                             const int lp2, const int rp2, const int lp3, const int rp3, const uint n) {
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uint r = i;
    const int i0 = r % (uint) ne0; r /= (uint) ne0;
    const int i1 = r % (uint) ne1; r /= (uint) ne1;
    const int i2 = r % (uint) ne2;
    const int i3 = r / (uint) ne2;
    float v = 0.0f;
    if (i0 >= lp0 && i0 < ne0 - rp0 && i1 >= lp1 && i1 < ne1 - rp1 &&
        i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
        v = *((__global const float *) (src0 + off0 + (ulong) (i0 - lp0)*nb00 + (ulong) (i1 - lp1)*nb01 +
                                        (ulong) (i2 - lp2)*nb02 + (ulong) (i3 - lp3)*nb03));
    }
    ((__global float *) (dst + offd))[i] = v;
}

// RMS_NORM + MUL (fusion): dst = (x / sqrt(mean(x^2) + eps)) * w, w broadcast over the rows like ggml_mul (src1
// of the MUL). One work-group (power-of-two size) per row; dimension 0 contiguous. The sum is reduced as in
// kernel_rms_norm_f32, so the normalized values are the same.
__kernel void kernel_rms_norm_mul_f32(__global const char * src0, const ulong off0, __global const char * w, const ulong offw,
                                      __global char * dst, const ulong offd,
                                      const int ne00, const ulong nb01, const ulong nb02, const ulong nb03,
                                      const int new0, const int new1, const int new2, const int new3,
                                      const ulong nbw1, const ulong nbw2, const ulong nbw3,
                                      const ulong nb1, const ulong nb2, const ulong nb3, const float eps,
                                      __local float * sum) {
    const int i1  = get_group_id(0);
    const int i2  = get_group_id(1);
    const int i3  = get_group_id(2);
    const int lid = get_local_id(0);
    __global const float * x = (__global const float *) (src0 + off0 + i1*nb01 + i2*nb02 + i3*nb03);
    // the weight row: without the integer modulo when the weight is not broadcast (some compilers, e.g. Beignet,
    // gave wrong results with it in this kernel)
    const int j1 = new1 == 1 ? 0 : (i1 < new1 ? i1 : i1 % new1);
    const int j2 = new2 == 1 ? 0 : (i2 < new2 ? i2 : i2 % new2);
    const int j3 = new3 == 1 ? 0 : (i3 < new3 ? i3 : i3 % new3);
    __global const float * y = (__global const float *) (w + offw + (ulong) j1*nbw1 + (ulong) j2*nbw2 + (ulong) j3*nbw3);
    __global       float * d = (__global       float *) (dst  + offd + i1*nb1  + i2*nb2  + i3*nb3);
    float s = 0.0f;
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        s += x[i0] * x[i0];
    }
    sum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (lid < k) {
            sum[lid] += sum[lid + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / sqrt(sum[0] / ne00 + eps);
    for (int i0 = lid; i0 < ne00; i0 += get_local_size(0)) {
        d[i0] = (x[i0] * scale) * y[new0 == ne00 ? i0 : i0 % new0];
    }
}

// TOP_K (F32 -> I32, k <= TOPK_MAX, indices in no particular order): one work-group (power-of-two size) per
// row. Every work-item keeps the k largest values of its part of the row in a list sorted in descending order;
// then k rounds pick the largest head of all the lists (ties: the lowest index) with a reduction in local memory.
#define TOPK_MAX 16
__kernel void kernel_top_k_f32(__global const char * src0, const ulong off0, const ulong nb01, const int ne00,
                               __global char * dst, const ulong offd, const ulong nb1, const int k,
                               __local float * lv, __local int * li) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float * x = (__global const float *) (src0 + off0 + (ulong) row*nb01);
    __global       int   * d = (__global       int   *) (dst  + offd + (ulong) row*nb1);

    float vals[TOPK_MAX];
    int   idxs[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; ++i) {
        vals[i] = -FLT_MAX;
        idxs[i] = -1;
    }
    for (int i = lid; i < ne00; i += lsz) {
        const float v = x[i];
        if (idxs[k - 1] >= 0 && v <= vals[k - 1]) {
            continue;
        }
        int p = k - 1;
        while (p > 0 && (idxs[p - 1] < 0 || v > vals[p - 1])) {
            vals[p] = vals[p - 1];
            idxs[p] = idxs[p - 1];
            --p;
        }
        vals[p] = v;
        idxs[p] = i;
    }

    int pos = 0; // first entry of the list not taken yet
    for (int r = 0; r < k; ++r) {
        lv[lid] = pos < k && idxs[pos] >= 0 ? vals[pos] : -FLT_MAX;
        li[lid] = pos < k ? idxs[pos] : -1;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz / 2; s > 0; s >>= 1) {
            if (lid < s) {
                const float va = lv[lid];
                const float vb = lv[lid + s];
                const int   ia = li[lid];
                const int   ib = li[lid + s];
                if (ib >= 0 && (ia < 0 || vb > va || (vb == va && ib < ia))) {
                    lv[lid] = vb;
                    li[lid] = ib;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const int best = li[0];
        if (lid == 0) {
            d[r] = best;
        }
        if (pos < k && idxs[pos] == best && best >= 0) {
            ++pos;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CLC";

// Flash attention kernel (see the comment inside); compiled per shape by ggml_cl_fa_kernel
static const char * be_fattn_source = R"CLC(
// FLASH_ATTN_EXT (F32 q, F16 or Q8_0 k/v, optional F16 mask) for OpenCL 1.1. Compiled for every shape with:
//   DK, DV   head sizes of K and V (DK % 64 == 0)
//   BQ, GH   a work-group computes R = BQ*GH rows: BQ consecutive query tokens x GH consecutive heads that
//            share their K/V head (so every K/V row is read once for all of them)
//   WG       work-items per group (128): 16 lanes per K row for the dot products (DK/16 values each, read
//            as half4, 16 lanes = 128 contiguous bytes), WG/16 K rows per pass, 64 K rows per tile
// Online softmax over the tiles of 64 K/V rows. Every row is computed with the same operations in the same
// order whatever BQ is and however many tokens the batch has, so a token gets the same result alone or in a
// batch (the verification of the multi-token prediction compares them); fully masked tiles change nothing
// (scale by exactly 1, add 0).
// A launch covers the query blocks qb0, qb0 + 1, ... (the host splits long operations, see be_fa_launch_macs).
//   SPLIT    (few query tokens: generation and verification batches) the K/V rows are split in chunks of
//            SPLIT tiles, one work-group per chunk, which writes the not normalized result of its rows with
//            their maximum and sum to part; kernel_flash_attn_combine merges the chunks of every row in order.
//            Without it, one token of Qwen3.5 ran on 2-4 work-groups whatever the context: the GT 720M was
//            almost idle with long contexts. The chunks have a fixed size (not derived from the number of
//            K/V rows, which depends on the padding of the cache), so the result of a row does not change
//            with the batch either. The program is built with -cl-fast-relaxed-math: no infinity may take part in
// the arithmetic, so masked positions (-INFINITY in the mask) are detected with a comparison and skipped.
//   K_Q8, V_Q8   K or V in Q8_0 (the cache with -ctk/-ctv q8_0), standard layout or q8r: k_b0/k_bh/k_nq and
//            v_b0/v_bh/v_nq as b0/bh/nq of be_q8kv_common (included before this source)
#define LANES  16
#define KROWS  (WG / LANES)
#define TILE   64
#define NPASS  (TILE / KROWS)
#define R      (BQ * GH)
#define NDV    ((DV + WG - 1) / WG)
#define NEG    (-FLT_MAX)
#ifdef SPLIT
#define SPLIT_ARGS , const int nh, __global float * part, const int nch
#else
#define SPLIT_ARGS
#endif

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void kernel_flash_attn_f16(
        __global const char * q, const ulong offq, const ulong nbq1, const ulong nbq2, const ulong nbq3,
        __global const char * k, const ulong offk, const ulong nbk1, const ulong nbk2, const ulong nbk3,
        __global const char * v, const ulong offv, const ulong nbv1, const ulong nbv2, const ulong nbv3,
        __global const char * mask, const ulong offm, const ulong nbm1, const ulong nbm2, const ulong nbm3,
        const int nem2, const int nem3, const int has_mask,
        __global char * dst, const ulong offd, const ulong nb1, const ulong nb2, const ulong nb3,
        const int N, const int KV, const int rk2, const int rk3, const int rv2, const int rv3,
        const float scale, const float softcap, const float max_bias, const float m0, const float m1, const int n_head_log2,
        const int k_b0, const int k_bh, const int k_nq, const int v_b0, const int v_bh, const int v_nq,
        const int qb0 SPLIT_ARGS) {
    __local float qs[R*DK];
    __local float red[R*KROWS*LANES];
    __local float ps[R*TILE];
    __local float rowm[R];
    __local float rows[R];
    __local float rowa[R];
    __local int   live;

    const int lid  = get_local_id(0);
    const int lane = lid % LANES;
    const int kr   = lid / LANES;
#ifdef SPLIT
    const int qb    = qb0 + get_group_id(0) / nch;
    const int chk   = get_group_id(0) % nch;
    const int t_beg = chk * (SPLIT*TILE);
    const int t_end = min(KV, t_beg + SPLIT*TILE);
#else
    const int qb    = qb0 + get_group_id(0);
    const int t_beg = 0;
    const int t_end = KV;
#endif
    const int h0   = get_group_id(1) * GH;
    const int iq3  = get_group_id(2);
    const int ik2  = h0 / rk2;
    const int iv2  = h0 / rv2;
    const int ik3  = iq3 / rk3;
    const int iv3  = iq3 / rv3;
#ifdef K_Q8
    const q8kv_t kl = q8kv_layout(k_b0, k_bh, k_nq, ik2);
#endif
#ifdef V_Q8
    const q8kv_t vl = q8kv_layout(v_b0, v_bh, v_nq, iv2);
#endif

    for (int i = lid; i < R*DK; i += WG) {
        const int r   = i / DK;
        const int d   = i % DK;
        const int iq1 = qb*BQ + r / GH;
        const int h   = h0 + r % GH;
        qs[i] = iq1 < N ? ((__global const float *) (q + offq + iq1*nbq1 + h*nbq2 + iq3*nbq3))[d] : 0.0f;
    }
    if (lid < R) {
        rowm[lid] = NEG;
        rows[lid] = 0.0f;
    }

    float acc[R*NDV];
    for (int i = 0; i < R*NDV; ++i) {
        acc[i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t0 = t_beg; t0 < t_end; t0 += TILE) {
        // a tile masked for every row of the group changes nothing: its K/V rows are not read (tlive = 0). Every
        // work-item still executes every barrier of the tile: some compilers, e.g. PoCL, mishandle barriers inside
        // conditional code or after a jump (skipping the tile with continue gave wrong results with Q8_0 K/V)
        if (lid == 0) {
            live = has_mask ? 0 : 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (has_mask) {
            for (int i = lid; i < R*TILE; i += WG) {
                const int r   = i / TILE;
                const int j   = t0 + i % TILE;
                const int iq1 = qb*BQ + r / GH;
                const int h   = h0 + r % GH;
                if (j < KV && iq1 < N) {
                    const float mv = vload_half(j, (__global const half *) (mask + offm + iq1*nbm1 + (h % nem2)*nbm2 + (iq3 % nem3)*nbm3));
                    if (mv >= -1e30f) {
                        live = 1;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef NOSKIP
        const int tlive = 1;
#else
        const int tlive = live; // read by every work-item before the next barrier, written again after it
#endif

        // 1) scores of the tile: NPASS passes of KROWS K rows, 16 lanes per row (in a dead tile every score is masked)
        for (int p = 0; p < NPASS; ++p) {
            const int jl = p*KROWS + kr;
            const int j  = t0 + jl;
            float dot[R];
            for (int r = 0; r < R; ++r) {
                dot[r] = 0.0f;
            }
            if (tlive && j < KV) {
                __global const char * kj = k + offk + j*nbk1 + ik2*nbk2 + ik3*nbk3;
                for (int s = 0; s < DK/(4*LANES); ++s) {
                    const int    d  = (s*LANES + lane)*4;
#ifdef K_Q8
                    const float4 kk = q8kv_get4(kj, kl, d);
#else
                    const float4 kk = vload_half4(0, (__global const half *) kj + d);
#endif
                    for (int r = 0; r < R; ++r) {
                        dot[r] += qs[r*DK + d]*kk.x + qs[r*DK + d + 1]*kk.y + qs[r*DK + d + 2]*kk.z + qs[r*DK + d + 3]*kk.w;
                    }
                }
            }
            for (int r = 0; r < R; ++r) {
                red[(r*KROWS + kr)*LANES + lane] = dot[r];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = lid; i < R*KROWS; i += WG) {
                const int r   = i / KROWS;
                const int kk  = i % KROWS;
                const int jl2 = p*KROWS + kk;
                const int j2  = t0 + jl2;
                const int iq1 = qb*BQ + r / GH;
                const int h   = h0 + r % GH;
                float sum = 0.0f;
                for (int l = 0; l < LANES; ++l) {
                    sum += red[(r*KROWS + kk)*LANES + l];
                }
                float sc = NEG;
                if (j2 < KV && iq1 < N) {
                    float mv = 0.0f;
                    bool  ok = true;
                    if (has_mask) {
                        mv = vload_half(j2, (__global const half *) (mask + offm + iq1*nbm1 + (h % nem2)*nbm2 + (iq3 % nem3)*nbm3));
                        ok = mv >= -1e30f;
                    }
                    if (ok) {
                        float slope = 1.0f;
                        if (max_bias > 0.0f) {
                            slope = h < n_head_log2 ? pow(m0, (float) (h + 1)) : pow(m1, (float) (2*(h - n_head_log2) + 1));
                        }
                        float sv = sum*scale;
                        if (softcap != 0.0f) {
                            sv = softcap*tanh(sv);
                        }
                        sc = sv + slope*mv;
                    }
                }
                ps[r*TILE + jl2] = sc;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // 2) online softmax of every row: new maximum, rescaling factor, probabilities and their sum
        for (int r = lid; r < R; r += WG) {
            float mx = NEG;
            for (int jl = 0; jl < TILE; ++jl) {
                mx = max(mx, ps[r*TILE + jl]);
            }
            const float mold = rowm[r];
            const float mnew = max(mold, mx);
            const float a    = mnew == mold ? 1.0f : exp(mold - mnew);
            float sum = 0.0f;
            for (int jl = 0; jl < TILE; ++jl) {
                const float sc = ps[r*TILE + jl];
                const float pv = sc > NEG ? exp(sc - mnew) : 0.0f;
                ps[r*TILE + jl] = pv;
                sum += pv;
            }
            rows[r] = rows[r]*a + sum;
            rowm[r] = mnew;
            rowa[r] = a;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3) V: every work-item accumulates dims lid, lid + WG, ... of every row
        for (int r = 0; r < R; ++r) {
            const float a = rowa[r];
            for (int i = 0; i < NDV; ++i) {
                acc[r*NDV + i] *= a;
            }
        }
        const int jn = tlive ? min(TILE, KV - t0) : 0;
        for (int jl = 0; jl < jn; ++jl) {
            __global const char * vj = v + offv + (t0 + jl)*nbv1 + iv2*nbv2 + iv3*nbv3;
            float vv[NDV];
            for (int i = 0; i < NDV; ++i) {
                const int d = lid + i*WG;
#ifdef V_Q8
                vv[i] = d < DV ? q8kv_get(vj, vl, d) : 0.0f;
#else
                vv[i] = d < DV ? vload_half(d, (__global const half *) vj) : 0.0f;
#endif
            }
            for (int r = 0; r < R; ++r) {
                const float pv = ps[r*TILE + jl];
                for (int i = 0; i < NDV; ++i) {
                    acc[r*NDV + i] += pv*vv[i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#ifdef SPLIT
    // part: for every row (iq3, iq1, h) and chunk, the DV values of the not normalized result, its maximum and
    // its sum (a chunk without visible K/V rows: 0, NEG and 0)
    for (int r = 0; r < R; ++r) {
        const int iq1 = qb*BQ + r / GH;
        const int h   = h0 + r % GH;
        if (iq1 >= N) {
            continue;
        }
        __global float * p = part + ((size_t) ((iq3*N + iq1)*nh + h)*nch + chk)*(DV + 2);
        for (int i = 0; i < NDV; ++i) {
            const int d = lid + i*WG;
            if (d < DV) {
                p[d] = acc[r*NDV + i];
            }
        }
        if (lid == 0) {
            p[DV]     = rowm[r];
            p[DV + 1] = rows[r];
        }
    }
#else
    // dst [DV, H, N, ne3]: element (d, h, iq1, iq3)
    for (int r = 0; r < R; ++r) {
        const int iq1 = qb*BQ + r / GH;
        const int h   = h0 + r % GH;
        if (iq1 >= N) {
            continue;
        }
        const float s   = rows[r];
        const float inv = s > 0.0f ? 1.0f/s : 0.0f;
        __global float * d0 = (__global float *) (dst + offd + h*nb1 + iq1*nb2 + iq3*nb3);
        for (int i = 0; i < NDV; ++i) {
            const int d = lid + i*WG;
            if (d < DV) {
                d0[d] = acc[r*NDV + i]*inv;
            }
        }
    }
#endif
}

#ifdef SPLIT
// merges the nch chunks of kernel_flash_attn_f16 (SPLIT) of every row in order: one work-group per row (iq3,
// iq1, h), its work-items over the DV values. Chunks without visible K/V rows weigh 0.
__kernel void kernel_flash_attn_combine(__global const float * part, const int nch,
        __global char * dst, const ulong offd, const ulong nb1, const ulong nb2, const ulong nb3,
        const int N, const int nh) {
    const int row = get_group_id(0);
    const int h   = row % nh;
    const int iq1 = (row / nh) % N;
    const int iq3 = row / (nh*N);
    __global const float * p = part + (size_t) row*nch*(DV + 2);
    float m = NEG;
    for (int c = 0; c < nch; ++c) {
        m = max(m, p[(size_t) c*(DV + 2) + DV]);
    }
    float l = 0.0f;
    for (int c = 0; c < nch; ++c) {
        const float mc = p[(size_t) c*(DV + 2) + DV];
        const float w  = mc == m ? 1.0f : (mc > NEG ? exp(mc - m) : 0.0f);
        l += w*p[(size_t) c*(DV + 2) + DV + 1];
    }
    const float inv = l > 0.0f ? 1.0f/l : 0.0f;
    __global float * d0 = (__global float *) (dst + offd + h*nb1 + iq1*nb2 + iq3*nb3);
    for (int d = get_local_id(0); d < DV; d += get_local_size(0)) {
        float a = 0.0f;
        for (int c = 0; c < nch; ++c) {
            const float mc = p[(size_t) c*(DV + 2) + DV];
            const float w  = mc == m ? 1.0f : (mc > NEG ? exp(mc - m) : 0.0f);
            a += w*p[(size_t) c*(DV + 2) + d];
        }
        d0[d] = a*inv;
    }
}
#endif
)CLC";

static std::string generate_kernels() {
    std::stringstream src;
    src << program_source << '\n';
    src << k_quants_source << '\n';
    for (size_t i = 0; i < dequant_str_values.size(); i += dequant_str_keys.size()) {
        std::string dequant_kernel = dequant_template;
        std::string dmmv_kernel = dequant_mul_mat_vec_template;
        for (size_t j = 0; j < dequant_str_keys.size(); j++) {
            replace(dequant_kernel, dequant_str_keys[j], dequant_str_values[i + j]);
            replace(dmmv_kernel, dequant_str_keys[j], dequant_mul_mat_vec_str_values[i + j]);
        }
        src << dequant_kernel << '\n';
        src << dmmv_kernel << '\n';
    }
    for (size_t i = 0; i < mul_str_values.size(); i += mul_str_keys.size()) {
        std::string mul_kernel = mul_template;
        for (size_t j = 0; j < mul_str_keys.size(); j++) {
            replace(mul_kernel, mul_str_keys[j], mul_str_values[i + j]);
        }
        src << mul_kernel << '\n';
    }
    src << add_template << '\n';
    src << dmmv_kq_l8_source << '\n';
    src << backend_ops_source << '\n';
    src << be_q6r_common << '\n';
    src << be_q6r_source << '\n';
    src << be_q45_source << '\n';
    src << be_attn_source << '\n';
    src << be_q8kv_common << '\n';
    src << be_modern_source << '\n';

    return src.str();
}

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_program program;
// the standard-layout Q4_K/Q5_K/Q6_K matrix x vector kernels (ggml_get_dmmv_l8_cl); the other kernels of
// program_source (the first OpenCL kernels of llama.cpp) are not created, bench-dmmv builds them to compare
static cl_kernel dequantize_mul_mat_vec_q4_K_l8_cl, dequantize_mul_mat_vec_q5_K_l8_cl, dequantize_mul_mat_vec_q6_K_l8_cl, dequantize_mul_mat_vec_q6_K_l8_ly_cl;

// builds an OpenCL program with the backend options; on a compile error it prints the log and returns
// nullptr (used for the optional programs of the backend, which are disabled if they do not build)
static cl_program try_build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
    cl_program p;
    char *program_log;
    size_t program_size;
    size_t log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        fprintf(stderr, "OpenCL error creating program");
        exit(1);
    }

    std::string compile_opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math "
                               "-DQK4_0=32 -DQR4_0=2 -DQK4_1=32 -DQR4_1=2 -DQK5_0=32 -DQR5_0=2 -DQK5_1=32 -DQR5_1=2 -DQK8_0=32 -DQR8_0=1 "
                               "-DQK_K=256 -DK_QUANTS_PER_ITERATION=" + std::to_string(K_QUANTS_PER_ITERATION);
    // GGML_OPENCL_CL_STD=CL1.1 forces the OpenCL C version (portability tests with strict compilers);
    // by default every device compiles with its highest OpenCL C 1.x version
    const char * cl_std = getenv("GGML_OPENCL_CL_STD");
    if (cl_std != nullptr && cl_std[0] != '\0') {
        compile_opts += std::string(" -cl-std=") + cl_std;
    }

    err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
    if(err < 0) {

        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        fprintf(stderr, "ggml_opencl: kernel compile error:\n\n%s\n", program_log);
        free(program_log);
        clReleaseProgram(p);
        return nullptr;
    }

    return p;
}

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
    cl_program p = try_build_program_from_source(ctx, dev, program_buffer);
    if (p == nullptr) {
        exit(1);
    }
    return p;
}

// ---- limits of the device (portability: the backend must work on any GPU with OpenCL 1.1) ----
// GGML_OPENCL_TEST_MAX_WG, GGML_OPENCL_TEST_LOCAL_MEM and GGML_OPENCL_TEST_MAX_ALLOC lower the work-group
// size, local memory and allocation limits of the device, to test the fallback paths on a GPU with
// higher limits.
static size_t ggml_cl_env_size(const char * name, size_t def) {
    const char * v = getenv(name);
    return v != nullptr && v[0] != '\0' ? (size_t) strtoull(v, nullptr, 10) : def;
}

// largest work-group size that kernel k can be launched with (it depends on its registers and on the device)
static size_t ggml_cl_kernel_max_wg(cl_kernel k) {
    static std::map<cl_kernel, size_t> cache;
    const auto it = cache.find(k);
    if (it != cache.end()) {
        return it->second;
    }
    size_t kmax = 0;
    size_t dmax = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(k, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax), &kmax, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(dmax), &dmax, NULL));
    kmax = std::min(kmax, dmax);
    kmax = std::min(kmax, ggml_cl_env_size("GGML_OPENCL_TEST_MAX_WG", kmax));
    cache[k] = kmax;
    return kmax;
}

// work-group size for a kernel that adapts to any size: the largest power of two <= preferred that k
// can be launched with (rms_norm and soft_max need a power of two for their reductions)
static size_t ggml_cl_wg(cl_kernel k, size_t preferred) {
    const size_t kmax = ggml_cl_kernel_max_wg(k);
    size_t wg = 1;
    while (wg * 2 <= preferred && wg * 2 <= kmax) {
        wg *= 2;
    }
    return wg;
}

static cl_ulong ggml_cl_local_mem(void) {
    static cl_ulong lmem = 0;
    if (lmem == 0) {
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL));
        lmem = std::min<cl_ulong>(lmem, ggml_cl_env_size("GGML_OPENCL_TEST_LOCAL_MEM", (size_t) lmem));
    }
    return lmem;
}

static cl_ulong ggml_cl_max_alloc(void) {
    static cl_ulong max_alloc = 0;
    if (max_alloc == 0) {
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL));
        max_alloc = std::min<cl_ulong>(max_alloc, ggml_cl_env_size("GGML_OPENCL_TEST_MAX_ALLOC", (size_t) max_alloc));
    }
    return max_alloc;
}

// simulated video memory (GGML_OPENCL_TEST_MAX_VRAM, in bytes, or with a K, M or G suffix): the buffers that the
// backend creates (weights, KV cache, compute buffers and its temporary buffers) are counted, and the
// allocation that goes beyond the limit fails as on a GPU with that memory. With GGML_OPENCL_TEST_MAX_ALLOC it
// lets a bigger device (or PoCL) check which models fit a smaller GPU. 0: no limit.
static size_t be_vram_used = 0;

static size_t ggml_cl_vram_limit(void) {
    static size_t limit = (size_t) -1;
    if (limit == (size_t) -1) {
        limit = 0;
        const char * v = getenv("GGML_OPENCL_TEST_MAX_VRAM");
        if (v != nullptr && v[0] != '\0') {
            char * end = nullptr;
            double x = strtod(v, &end);
            if (end != nullptr && (*end == 'K' || *end == 'k')) { x *= 1024.0; }
            if (end != nullptr && (*end == 'M' || *end == 'm')) { x *= 1024.0*1024.0; }
            if (end != nullptr && (*end == 'G' || *end == 'g')) { x *= 1024.0*1024.0*1024.0; }
            limit = (size_t) x;
        }
    }
    return limit;
}

// counts size bytes of video memory for what; false (and a message) if they go beyond the simulated limit
static bool ggml_cl_vram_take(size_t size, const char * what) {
    const size_t limit = ggml_cl_vram_limit();
    if (limit > 0 && be_vram_used + size > limit) {
        fprintf(stderr, "ggml_opencl: %s: %.2f MiB do not fit the simulated video memory (GGML_OPENCL_TEST_MAX_VRAM = %.2f MiB, %.2f MiB in use)\n",
                what, size/1024.0/1024.0, limit/1024.0/1024.0, be_vram_used/1024.0/1024.0);
        return false;
    }
    be_vram_used += size;
    return true;
}

static void ggml_cl_vram_give(size_t size) {
    be_vram_used -= std::min(size, be_vram_used);
}

// true if kernel k can be launched with work-groups of exactly wg work-items and dyn_local bytes of
// dynamic local memory (call it before setting the local memory arguments of k: CL_KERNEL_LOCAL_MEM_SIZE
// includes the ones already set)
static bool ggml_cl_kernel_fits(cl_kernel k, size_t wg, size_t dyn_local) {
    if (k == nullptr) {
        return false;
    }
    cl_ulong lmem = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(k, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL));
    return ggml_cl_kernel_max_wg(k) >= wg && lmem + dyn_local <= ggml_cl_local_mem();
}

// ---- devices ----
// The registry (ggml_backend_opencl_reg) exposes the OpenCL devices found when it is created: every GPU or
// accelerator of every platform or, if GGML_OPENCL_PLATFORM and/or GGML_OPENCL_DEVICE are set, only the
// devices that match them (a number selects by index, any other text by part of the name; then devices of
// any type are accepted, e.g. PoCL on the CPU for testing). The backend keeps its state (context, programs,
// kernels, queue and tuning) in globals, so a process uses a single device: the first one that is used
// becomes the active device, and using another one fails with a message.

struct ggml_cl_device_info {
    cl_platform_id platform;
    cl_device_id   device;
    std::string    platform_name;
    std::string    name;
    std::string    version;
    cl_device_type type;
    bool           unified_memory; // CL_DEVICE_HOST_UNIFIED_MEMORY (integrated GPU that uses the RAM)
    size_t         global_mem;
};

static std::vector<ggml_cl_device_info> g_cl_devices;
static int g_cl_active = -1; // index in g_cl_devices of the device in use, -1 before the first use
// every kernel is flushed on its own (clFlush after each launch). Beignet (Intel HD Graphics 2500/4000 and
// Haswell) does not always make the results of a kernel visible to the next kernel of the in-order queue, as
// OpenCL requires: chains of operations gave wrong results now and then on the HD 4000, and a flush after every
// kernel fixes them (generation 5% slower there; a barrier, 23%). On by default with Beignet;
// GGML_OPENCL_KERNEL_FLUSH=0|1 forces it off or on.
static bool g_cl_kernel_flush = false;

static std::string ggml_cl_device_string(cl_device_id dev, cl_device_info param) {
    size_t size = 0;
    CL_CHECK(clGetDeviceInfo(dev, param, 0, NULL, &size));
    std::string s(size, '\0');
    CL_CHECK(clGetDeviceInfo(dev, param, size, &s[0], NULL));
    s.resize(strlen(s.c_str()));
    return s;
}

static std::string ggml_cl_platform_string(cl_platform_id p, cl_platform_info param) {
    size_t size = 0;
    CL_CHECK(clGetPlatformInfo(p, param, 0, NULL, &size));
    std::string s(size, '\0');
    CL_CHECK(clGetPlatformInfo(p, param, size, &s[0], NULL));
    s.resize(strlen(s.c_str()));
    return s;
}

// requirements of the device: OpenCL 1.1 or newer (byte-addressable stores and the OpenCL C 1.1
// built-ins are core since 1.1), little-endian (the kernels reinterpret 32-bit words as bytes) and
// 64-bit integers (the byte offsets are ulong; optional in the embedded profile). A device that does not
// meet them is skipped with a message.
static bool ggml_cl_device_ok(cl_device_id dev, const std::string & name) {
    const std::string version = ggml_cl_device_string(dev, CL_DEVICE_VERSION);
    const std::string profile = ggml_cl_device_string(dev, CL_DEVICE_PROFILE);
    const std::string exts    = ggml_cl_device_string(dev, CL_DEVICE_EXTENSIONS);
    cl_bool little = CL_FALSE;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_ENDIAN_LITTLE, sizeof(little), &little, NULL));
    int major = 0;
    int minor = 0;
    if (sscanf(version.c_str(), "OpenCL %d.%d", &major, &minor) != 2 || major < 1 || (major == 1 && minor < 1)) {
        fprintf(stderr, "ggml_opencl: skipping device '%s': it reports '%s'; OpenCL 1.1 or newer is required\n", name.c_str(), version.c_str());
        return false;
    }
    if (little != CL_TRUE) {
        fprintf(stderr, "ggml_opencl: skipping device '%s': it is big-endian; a little-endian device is required\n", name.c_str());
        return false;
    }
    if (profile == "EMBEDDED_PROFILE" && exts.find("cles_khr_int64") == std::string::npos) {
        fprintf(stderr, "ggml_opencl: skipping device '%s': embedded profile without 64-bit integers (cles_khr_int64)\n", name.c_str());
        return false;
    }
    return true;
}

// true if want (from GGML_OPENCL_PLATFORM or GGML_OPENCL_DEVICE) selects the platform or device with this
// index and names: no value selects everything, a number selects by index, any other text by part of a name
static bool ggml_cl_matches(const char * want, unsigned index, const std::string & a, const std::string & b) {
    if (want == nullptr || want[0] == '\0') {
        return true;
    }
    char * end = nullptr;
    const unsigned long n = strtoul(want, &end, 10);
    if (end != want && *end == '\0') {
        return n == index;
    }
    return a.find(want) != std::string::npos || b.find(want) != std::string::npos;
}

static void ggml_cl_probe_devices(void) {
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        fprintf(stderr, "ggml_opencl: no OpenCL platform found\n");
        return;
    }
    std::vector<cl_platform_id> pids(n_platforms);
    CL_CHECK(clGetPlatformIDs(n_platforms, pids.data(), NULL));

    const char * want_p = getenv("GGML_OPENCL_PLATFORM");
    const char * want_d = getenv("GGML_OPENCL_DEVICE");
    const bool   chosen = (want_p != nullptr && want_p[0] != '\0') || (want_d != nullptr && want_d[0] != '\0');

    unsigned dev_index = 0; // index among the devices of the platforms that match GGML_OPENCL_PLATFORM
    for (cl_uint ip = 0; ip < n_platforms; ++ip) {
        const std::string pname   = ggml_cl_platform_string(pids[ip], CL_PLATFORM_NAME);
        const std::string pvendor = ggml_cl_platform_string(pids[ip], CL_PLATFORM_VENDOR);
        if (!ggml_cl_matches(want_p, ip, pname, pvendor)) {
            continue;
        }
        cl_uint n_dev = 0;
        if (clGetDeviceIDs(pids[ip], CL_DEVICE_TYPE_ALL, 0, NULL, &n_dev) != CL_SUCCESS || n_dev == 0) {
            continue;
        }
        std::vector<cl_device_id> dids(n_dev);
        CL_CHECK(clGetDeviceIDs(pids[ip], CL_DEVICE_TYPE_ALL, n_dev, dids.data(), NULL));
        for (cl_uint id = 0; id < n_dev; ++id, ++dev_index) {
            const std::string dname = ggml_cl_device_string(dids[id], CL_DEVICE_NAME);
            cl_device_type type = 0;
            CL_CHECK(clGetDeviceInfo(dids[id], CL_DEVICE_TYPE, sizeof(type), &type, NULL));
            if (!ggml_cl_matches(want_d, dev_index, dname, dname)) {
                continue;
            }
            // OpenCL implementations on the CPU (PoCL, ...) only when they are selected explicitly
            if (!chosen && (type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)) == 0) {
                continue;
            }
            if (!ggml_cl_device_ok(dids[id], dname)) {
                continue;
            }
            ggml_cl_device_info info;
            info.platform      = pids[ip];
            info.device        = dids[id];
            info.platform_name = pname;
            info.name          = dname;
            info.version       = ggml_cl_device_string(dids[id], CL_DEVICE_VERSION);
            info.type          = type;
            cl_bool  unified = CL_FALSE;
            cl_ulong gmem    = 0;
            CL_CHECK(clGetDeviceInfo(dids[id], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL));
            CL_CHECK(clGetDeviceInfo(dids[id], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL));
            info.unified_memory = unified == CL_TRUE;
            info.global_mem     = (size_t) gmem;
            g_cl_devices.push_back(info);
        }
    }
    if (chosen && g_cl_devices.empty()) {
        fprintf(stderr, "ggml_opencl: no OpenCL device matches GGML_OPENCL_PLATFORM='%s' and GGML_OPENCL_DEVICE='%s'\n",
                want_p ? want_p : "", want_d ? want_d : "");
    }
}

static void ggml_cl_be_init(void);

// creates the context, builds the programs and creates the kernels for device idx of g_cl_devices (once
// per process); false (and a message) if another device is already in use
static bool ggml_cl_activate(int idx) {
    if (g_cl_active == idx) {
        return true;
    }
    if (g_cl_active >= 0) {
        fprintf(stderr, "ggml_opencl: OpenCL%d (%s) cannot be used: the backend uses a single OpenCL device per process and "
                        "OpenCL%d (%s) is in use; choose one device with -dev or GGML_OPENCL_PLATFORM/GGML_OPENCL_DEVICE\n",
                idx, g_cl_devices[idx].name.c_str(), g_cl_active, g_cl_devices[g_cl_active].name.c_str());
        return false;
    }
    const ggml_cl_device_info & info = g_cl_devices[idx];

    fprintf(stderr, "ggml_opencl: selecting platform: '%s'\n", info.platform_name.c_str());
    fprintf(stderr, "ggml_opencl: selecting device: '%s'\n", info.name.c_str());
    fprintf(stderr, "ggml_opencl: device version: '%s'\n", info.version.c_str());
    if ((info.type & CL_DEVICE_TYPE_GPU) == 0) {
        fprintf(stderr, "ggml_opencl: warning, not a GPU: '%s'.\n", info.name.c_str());
    }
    const char * kernel_flush = getenv("GGML_OPENCL_KERNEL_FLUSH");
    g_cl_kernel_flush = kernel_flush != nullptr ? atoi(kernel_flush) != 0 :
                                                  info.platform_name.find("Intel Gen OCL") != std::string::npos;
    if (g_cl_kernel_flush) {
        fprintf(stderr, "ggml_opencl: every kernel is flushed on its own (%s)\n",
                kernel_flush != nullptr ? "GGML_OPENCL_KERNEL_FLUSH" : "Beignet: results of a kernel not visible to the next one otherwise");
    }

    platform = info.platform;
    device   = info.device;

    cl_int err;
    cl_context_properties properties[] = {
        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)platform, 0
    };

    CL_CHECK((context = clCreateContext(properties, 1, &device, NULL, NULL, &err), err));

    const std::string kernel_src = generate_kernels();

    program = build_program_from_source(context, device, kernel_src.c_str());

    CL_CHECK((dequantize_mul_mat_vec_q4_K_l8_cl    = clCreateKernel(program, "dequantize_mul_mat_vec_q4_K_l8",    &err), err));
    CL_CHECK((dequantize_mul_mat_vec_q5_K_l8_cl    = clCreateKernel(program, "dequantize_mul_mat_vec_q5_K_l8",    &err), err));
    CL_CHECK((dequantize_mul_mat_vec_q6_K_l8_cl    = clCreateKernel(program, "dequantize_mul_mat_vec_q6_K_l8",    &err), err));
    CL_CHECK((dequantize_mul_mat_vec_q6_K_l8_ly_cl = clCreateKernel(program, "dequantize_mul_mat_vec_q6_K_l8_ly", &err), err));

    g_cl_active = idx;
    ggml_cl_be_init();
    return true;
}

// fast dequantize_mul_mat_vec kernels (8 lanes per row) for Q4_K, Q5_K and Q6_K; returns nullptr for
// other types. local_size is the work-group size and local_y is true if the kernel takes an extra
// __local float[ncols] argument. Configurations measured with examples/bench-dmmv on a GT 720M.
// work-groups of 128 (Q4_K, Q5_K, Q6_K with y in global memory), or of 256 with y staged in local
// memory (Q6_K, rows of up to 2048 columns); devices that cannot run those sizes get the largest power of
// two they can (the kernels work with any multiple of 8: 8 lanes per row)
static cl_kernel* ggml_get_dmmv_l8_cl(ggml_type type, int64_t ncols, size_t * local_size, bool * local_y) {
    *local_y = false;
    cl_kernel * k = nullptr;
    switch (type) {
        case GGML_TYPE_Q4_K:
            k = &dequantize_mul_mat_vec_q4_K_l8_cl;
            break;
        case GGML_TYPE_Q5_K:
            k = &dequantize_mul_mat_vec_q5_K_l8_cl;
            break;
        case GGML_TYPE_Q6_K: {
            // decided once, before the local memory arguments of the kernel are ever set
            static const bool ly_ok = ggml_cl_kernel_fits(dequantize_mul_mat_vec_q6_K_l8_ly_cl, 256, (256 + 2048) * sizeof(float));
            if (ncols <= 2048 && ly_ok) {
                *local_size = 256;
                *local_y    = true;
                return &dequantize_mul_mat_vec_q6_K_l8_ly_cl;
            }
            k = &dequantize_mul_mat_vec_q6_K_l8_cl;
            break;
        }
        default:
            return nullptr;
    }
    *local_size = ggml_cl_wg(*k, 128);
    if (*local_size < 8) {
        fprintf(stderr, "ggml_opencl: the matrix x vector kernel needs work-groups of at least 8, the device allows %zu\n", *local_size);
        exit(1);
    }
    return k;
}

// ggml-backend
//
// Complete OpenCL backend (ggml_backend_opencl). Tensors live in OpenCL buffers; the data pointer of
// a tensor is cl_ptr_base + byte offset inside its buffer, so every kernel receives (cl_mem, offset).
// All buffer transfers and kernels go through one in-order queue (be_queue), so the operations of a
// graph run in order without events. ggml_backend_sched sends to this backend every operation that
// ggml_backend_opencl_supports_op accepts; the rest runs on the CPU backend with copies.

static void * const cl_ptr_base = (void *)(uintptr_t) 0x1000;

static cl_command_queue be_queue = nullptr; // in-order queue of the backend

// The kernels, the queue, the temporary buffers and the simulated video memory are shared by every context of the
// process, so the entry points that use them take this lock: the work of one graph is enqueued without the
// commands of another one in between (several contexts in different threads, e.g. test-thread-safety), and
// clSetKernelArg is never called on the same kernel from two threads. Recursive: an entry point may call another.
static std::recursive_mutex be_mutex;

// creates a device buffer of size bytes and makes the driver allocate it now: some drivers (NVIDIA) only
// allocate a buffer when a command first uses it, so a buffer that does not fit made the first transfer
// fail with CL_MEM_OBJECT_ALLOCATION_FAILURE (error -4) and abort the program. A one-byte write at its end
// moves that failure here. nullptr (and a message) if it does not fit, also in the simulated video memory
// (GGML_OPENCL_TEST_MAX_VRAM).
static cl_mem ggml_cl_create_buffer_now(size_t size, const char * what) {
    size = std::max<size_t>(size, 1);
    if (!ggml_cl_vram_take(size, what)) {
        return nullptr;
    }
    cl_int err;
    cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err == CL_SUCCESS) {
        const uint8_t zero = 0;
        err = clEnqueueWriteBuffer(be_queue, mem, CL_TRUE, size - 1, 1, &zero, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(mem);
        }
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ggml_opencl: %s: %.2f MiB do not fit the device (error %d)\n", what, size/1024.0/1024.0, (int) err);
        ggml_cl_vram_give(size);
        return nullptr;
    }
    return mem;
}

struct ggml_cl_be_kernels {
    cl_kernel add, mul, scale, silu, rms_norm, cpy, get_rows, soft_max, rope, mul_mat_f, mul_mat_f_tiled;
    cl_kernel sigmoid, softplus, concat, ssm_conv1d, gated_delta_net; // qwen35
    cl_kernel set_rows, swiglu, l2_norm, arange, pad; // operations of the current ggml graphs
    cl_kernel rms_norm_mul; // fusions
    cl_kernel top_k;
    cl_kernel get_rows_q4_K, get_rows_q5_K, get_rows_q6_K;
    cl_kernel dequant_q4_K, dequant_q5_K, dequant_q6_K;
    cl_kernel gemm_n64, gemm_n32; // kernel_gemm_f32 with 128x64 and 256x32 tiles
    cl_kernel mmvn_pack[3], mmvn_q4_K[3], mmvn_q5_K[3], mmvn_q6_K[3]; // be_mmvn_source with NC = be_mmvn_nc[i]
    cl_kernel dmmv_q6r, dmmv_q6r_v2, dequant_q6r, get_rows_q6r, mmvn_q6r[3]; // q6r layout (be_q6r_source)
    cl_kernel dmmv_q4_K_ly1, dmmv_q4_K_lyc, dmmv_q5_K_ly1, dmmv_q5_K_lyc; // be_q45_source (y in one chunk / in chunks)
    cl_kernel dmmvn_q4_K[3], dmmvn_q5_K[3], dmmvn_q6r[3], dmmvn_q6r_v2[3]; // be_dmmvn_source with NC = 2 + i
    cl_kernel dmmv_q8r[4], dmmv_q8r_v2[4], get_rows_q8r, dequant_q8r;   // be_q8_source with NC = 1 + i (q8r layout)
    cl_kernel mul_mat_vec_f, ssm_conv1d_tiled, argmax; // be_attn_source
    cl_kernel transpose_f16, set_rows_q8_0, dequant_q8_0_kv; // decomposed flash attention, Q8_0 K/V cache
};
static ggml_cl_be_kernels be_k;

// Q6_K weights with an even number of super-blocks per row (ne[0] % 512 == 0, rows 4-byte aligned) are
// kept in the OpenCL buffers in the q6r layout (see be_q6r_common): set_tensor reorders them, get_tensor
// restores block_q6_K, and every kernel that reads Q6_K weights has a q6r version. Decided in
// ggml_cl_be_init, before the first buffer exists: off with GGML_OPENCL_Q6R=0, or if kernel_dmmv_q6r does
// not fit the device. be_q6r_wg and be_q6r_ych: work-group size and chunk of y of kernel_dmmv_q6r.
static bool   be_q6r_on      = false;
static size_t be_q6r_wg      = 0;
static int    be_q6r_ych     = 0;
static bool   be_mmvn_q6r_ok = false; // kernel_mmvn_q6r (be_mmvn_lwg work-items)

// matrix x vector for Q4_K / Q5_K with y in local memory (be_q45_source): work-group size and chunk of
// y, decided in ggml_cl_be_init (off with GGML_OPENCL_Q45_LY=0 or if the kernels do not fit the device:
// then dequantize_mul_mat_vec_q4_K_l8 / q5_K_l8). be_cu: compute units of the device (choice of T).
static bool   be_q45_on  = false;
static size_t be_q45_wg  = 0;
static int    be_q45_ych = 0;
static int    be_cu      = 1;

// matrix x NC vectors with the tuned kernels (be_dmmvn_source), for the verification batches of
// multi-token prediction and prompts of 2 to be_dmmvn_max_cols tokens: work-group size and chunk of y
// staged in local memory of every NC (index NC - 2; sch 0: not compiled or does not fit the device, the
// batch takes the mmvn path). Decided in ggml_cl_be_init: chunks of at most 2048 values (and at most
// be_q45_ych for Q4_K/Q5_K) with NC*sch <= 4096, the largest that fits. Every column gets the operations of
// the matrix x vector kernel, so on the GT 720M the verification batches of the multi-token prediction
// (up to --mtp 3) give bit for bit the logits of decoding their tokens one by one and MTP never changes
// the output (checked with whole batches of Qwen3.5-2B; other compilers may reorder the operations, see
// be_dmmvn_source). Time per call against the mmvn kernels
// (GT 720M, Qwen3.5-2B): 2 columns: Q4_K +3%, Q5_K +3%, q6r -7%; 3 columns: about -12% for the three
// types; 4 columns: +4% to +10%. GGML_OPENCL_DMMVN_MAX_COLS: 2-4 columns at most (default 4), 0
// disables the path. Why it does not get closer to the cost of one column: the kernels are limited by
// instructions (the conversions of the quants run at a third of the rate of the FMAs on Fermi, and every
// column adds its FMAs and local loads), and NC = 2 takes 45-53 registers instead of 38-42 (2
// work-groups of 256 per multiprocessor instead of 3; holding fewer values alive saved 1-2 registers).
static int    be_dmmvn_max_cols   = 4;
static size_t be_dmmvn_q45_wg[3]  = { 0, 0, 0 };
static int    be_dmmvn_q45_sch[3] = { 0, 0, 0 };
static size_t be_dmmvn_q6r_wg[3]  = { 0, 0, 0 };
static int    be_dmmvn_q6r_sch[3] = { 0, 0, 0 };

// Q8_0 weights with an even number of blocks per row (ne[0] % 64 == 0) are kept in the OpenCL buffers in the
// q8r layout (see be_q8_source) when its kernels fit the device (decided in ggml_cl_be_init, before the first
// buffer exists; off with GGML_OPENCL_Q8R=0, and then Q8_0 operations run on the CPU). be_q8r_wg / be_q8r_sch:
// work-group size and chunk of y of kernel_dmmv_q8r for NC = 1 + i columns (sch 0: that NC does not fit, its
// columns are split into launches with fewer).
static bool   be_q8r_on      = false;
static size_t be_q8r_wg[4]   = { 0, 0, 0, 0 };
static int    be_q8r_sch[4]  = { 0, 0, 0, 0 };

// tile shapes of kernel_gemm_f32 (BM, BN, BK, TM, TN, PAD), from examples/bench-gemm on the GT 720M:
// 128x64 reaches 115-131 GFLOPS (50-56% of the measured FP32 peak of 234 GFLOPS) with 512 columns;
// 256x32 is the fastest one with 17-32 columns (81 GFLOPS with 32)
struct ggml_cl_be_gemm_shape {
    int bm, bn, bk, tm, tn, pad;
};
static const ggml_cl_be_gemm_shape be_gemm_n64 = { 128, 64, 16, 8, 4, 2 };
static const ggml_cl_be_gemm_shape be_gemm_n32 = { 256, 32, 16, 8, 4, 2 };

// optional profiling of the graph computation (environment variable GGML_OPENCL_PROFILE): the time is
// accumulated per operation (and per MUL_MAT path) and a table is printed when the backend is freed.
// GGML_OPENCL_PROFILE=1: every node is waited for and timed on the host (includes the launch and the wait,
// which inflates the small kernels); GGML_OPENCL_PROFILE=2: GPU time of every kernel from profiling
// events, without waiting per node, plus the idle time of the GPU between kernels inside each graph.
struct ggml_cl_be_prof_entry {
    double  ms    = 0.0;
    int64_t count = 0;
};
static bool                                         be_profile     = false;
static bool                                         be_sync        = false; // GGML_OPENCL_SYNC=1: wait for every node and name the one that fails
static bool                                         be_profile_gpu = false;
static std::map<std::string, ggml_cl_be_prof_entry> be_prof;
static const char *                                 be_mm_path = ""; // path taken by the last MUL_MAT
static std::vector<cl_event>                        be_node_events; // kernels of the current node (GGML_OPENCL_PROFILE=2)

// GEMM of the dequantized weights while processing a prompt (environment variable GGML_OPENCL_GEMM):
// kernel_gemm_f32 (default) or the generic tiled kernel kernel_mul_mat_f_tiled ("tiled"). On the GT 720M,
// kernel_gemm_f32 is 1.4-1.6x faster than the generic tiled kernel, which was 17% faster than the CLBlast
// library that the first OpenCL backend of llama.cpp used.
enum ggml_cl_be_gemm_kind {
    BE_GEMM_OWN,
    BE_GEMM_TILED,
};
static ggml_cl_be_gemm_kind be_gemm_kind = BE_GEMM_OWN;

// batches with more columns than this use dequantization + GEMM; the smaller ones, one matrix x
// vector product per column (environment variable GGML_OPENCL_GEMM_MIN_COLS). On the GT 720M both
// cost the same with 14 columns (Qwen2.5-1.5B: pp12 8.3 against 7.1 t/s, pp16 8.5 against 9.7 t/s).
static int64_t be_gemm_min_cols = 14;

// matrix x several vectors (be_mmvn_source) for batches of 2 to be_mmvn_max_cols columns: launches of
// NC = 8 columns, and the rest with NC = 2 (1-2 columns), 4 (3-4) or 8 (5-7). R = 2 rows per group of 8
// lanes, work-groups of 128. Measured on the GT 720M (examples/bench-gemm, BENCH_GEMM_MMVN=1), it beats
// dequantization + GEMM up to about 48 columns with Q4_K/Q5_K and 28 with Q6_K; with 8 columns it is
// 2.7-3.7 times faster than the matrix x vector kernel once per column. GGML_OPENCL_MMVN_MAX_COLS
// overrides both limits (0 disables the path).
static const int be_mmvn_nc[3] = { 2, 4, 8 };
static const int be_mmvn_r     = 2;
static const int be_mmvn_lwg   = 128;
static int64_t   be_mmvn_max_cols_q45 = 48;
static int64_t   be_mmvn_max_cols_q6  = 28;
static cl_mem    be_yi_buf  = nullptr; // packed columns (K*NC floats)
static size_t    be_yi_size = 0;

// temporary F32 buffer for the dequantized weights of the GEMM path (at most 256 MiB, and never more
// than the largest allocation of the device)
static cl_mem be_dq_buf  = nullptr;
static size_t be_dq_size = 0;
static size_t be_dq_max_size = 256u*1024u*1024u;

// paths whose kernels have a fixed work-group size (or a large local memory) and may not fit every
// device; decided in ggml_cl_be_init with the limits of the device (see ggml_cl_kernel_fits)
static bool be_gemm_n64_ok = false; // kernel_gemm_f32 128x64 (256 work-items)
static bool be_gemm_n32_ok = false; // kernel_gemm_f32 256x32 (256 work-items, ~18.7 KB of local memory)
static bool be_tiled_ok    = false; // kernel_mul_mat_f_tiled (16x16 work-items)
static bool be_mmvn_ok[3]  = { false, false, false }; // kernel_mmvn_* for Q4_K, Q5_K, Q6_K (be_mmvn_lwg work-items)

static void ggml_cl_be_init(void) {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    cl_int err;
    {
        const char * prof = getenv("GGML_OPENCL_PROFILE");
        be_profile     = prof != nullptr;
        be_profile_gpu = prof != nullptr && strcmp(prof, "2") == 0;
        be_sync        = getenv("GGML_OPENCL_SYNC") != nullptr && atoi(getenv("GGML_OPENCL_SYNC")) > 0;
    }
    CL_CHECK((be_queue = clCreateCommandQueue(context, device, be_profile_gpu ? CL_QUEUE_PROFILING_ENABLE : 0, &err), err));
    CL_CHECK((be_k.add          = clCreateKernel(program, "kernel_add_f32",      &err), err));
    CL_CHECK((be_k.mul          = clCreateKernel(program, "kernel_mul_f32",      &err), err));
    CL_CHECK((be_k.scale        = clCreateKernel(program, "kernel_scale_f32",    &err), err));
    CL_CHECK((be_k.silu         = clCreateKernel(program, "kernel_silu_f32",     &err), err));
    CL_CHECK((be_k.sigmoid      = clCreateKernel(program, "kernel_sigmoid_f32",  &err), err));
    CL_CHECK((be_k.softplus     = clCreateKernel(program, "kernel_softplus_f32", &err), err));
    CL_CHECK((be_k.concat       = clCreateKernel(program, "kernel_concat_f32",   &err), err));
    CL_CHECK((be_k.ssm_conv1d   = clCreateKernel(program, "kernel_ssm_conv1d_f32", &err), err));
    CL_CHECK((be_k.gated_delta_net = clCreateKernel(program, "kernel_gated_delta_net_f32", &err), err));
    CL_CHECK((be_k.rms_norm     = clCreateKernel(program, "kernel_rms_norm_f32", &err), err));
    CL_CHECK((be_k.cpy          = clCreateKernel(program, "kernel_cpy",          &err), err));
    CL_CHECK((be_k.get_rows     = clCreateKernel(program, "kernel_get_rows",     &err), err));
    CL_CHECK((be_k.soft_max     = clCreateKernel(program, "kernel_soft_max",     &err), err));
    CL_CHECK((be_k.rope         = clCreateKernel(program, "kernel_rope_f32",     &err), err));
    CL_CHECK((be_k.mul_mat_f    = clCreateKernel(program, "kernel_mul_mat_f",    &err), err));
    CL_CHECK((be_k.mul_mat_f_tiled = clCreateKernel(program, "kernel_mul_mat_f_tiled", &err), err));
    CL_CHECK((be_k.dequant_q4_K = clCreateKernel(program, "kernel_dequant_q4_K", &err), err));
    CL_CHECK((be_k.dequant_q5_K = clCreateKernel(program, "kernel_dequant_q5_K", &err), err));
    CL_CHECK((be_k.dequant_q6_K = clCreateKernel(program, "kernel_dequant_q6_K", &err), err));
    CL_CHECK((be_k.get_rows_q4_K = clCreateKernel(program, "kernel_get_rows_q4_K", &err), err));
    CL_CHECK((be_k.get_rows_q5_K = clCreateKernel(program, "kernel_get_rows_q5_K", &err), err));
    CL_CHECK((be_k.get_rows_q6_K = clCreateKernel(program, "kernel_get_rows_q6_K", &err), err));
    CL_CHECK((be_k.dmmv_q6r      = clCreateKernel(program, "kernel_dmmv_q6r",      &err), err));
    CL_CHECK((be_k.dmmv_q6r_v2   = clCreateKernel(program, "kernel_dmmv_q6r_v2",   &err), err));
    CL_CHECK((be_k.dequant_q6r   = clCreateKernel(program, "kernel_dequant_q6r",   &err), err));
    CL_CHECK((be_k.get_rows_q6r  = clCreateKernel(program, "kernel_get_rows_q6r",  &err), err));
    CL_CHECK((be_k.dmmv_q4_K_ly1 = clCreateKernel(program, "kernel_dmmv_q4_K_ly1", &err), err));
    CL_CHECK((be_k.dmmv_q4_K_lyc = clCreateKernel(program, "kernel_dmmv_q4_K_lyc", &err), err));
    CL_CHECK((be_k.dmmv_q5_K_ly1 = clCreateKernel(program, "kernel_dmmv_q5_K_ly1", &err), err));
    CL_CHECK((be_k.dmmv_q5_K_lyc = clCreateKernel(program, "kernel_dmmv_q5_K_lyc", &err), err));
    CL_CHECK((be_k.mul_mat_vec_f = clCreateKernel(program, "kernel_mul_mat_vec_f", &err), err));
    CL_CHECK((be_k.ssm_conv1d_tiled = clCreateKernel(program, "kernel_ssm_conv1d_tiled", &err), err));
    CL_CHECK((be_k.argmax        = clCreateKernel(program, "kernel_argmax_f32",    &err), err));
    CL_CHECK((be_k.set_rows      = clCreateKernel(program, "kernel_set_rows",      &err), err));
    CL_CHECK((be_k.swiglu        = clCreateKernel(program, "kernel_swiglu_f32",    &err), err));
    CL_CHECK((be_k.l2_norm       = clCreateKernel(program, "kernel_l2_norm_f32",   &err), err));
    CL_CHECK((be_k.arange        = clCreateKernel(program, "kernel_arange_f32",    &err), err));
    CL_CHECK((be_k.pad           = clCreateKernel(program, "kernel_pad_f32",       &err), err));
    CL_CHECK((be_k.rms_norm_mul  = clCreateKernel(program, "kernel_rms_norm_mul_f32", &err), err));
    CL_CHECK((be_k.top_k         = clCreateKernel(program, "kernel_top_k_f32",     &err), err));
    CL_CHECK((be_k.transpose_f16 = clCreateKernel(program, "kernel_transpose_f16", &err), err));
    CL_CHECK((be_k.set_rows_q8_0 = clCreateKernel(program, "kernel_set_rows_q8_0", &err), err));
    CL_CHECK((be_k.dequant_q8_0_kv = clCreateKernel(program, "kernel_dequant_q8_0_kv", &err), err));
    for (const ggml_cl_be_gemm_shape * s : { &be_gemm_n64, &be_gemm_n32 }) {
        const std::string src = "#define BM " + std::to_string(s->bm) + "\n#define BN " + std::to_string(s->bn) +
                                "\n#define BK " + std::to_string(s->bk) + "\n#define TM " + std::to_string(s->tm) +
                                "\n#define TN " + std::to_string(s->tn) + "\n#define PAD " + std::to_string(s->pad) + "\n" +
                                be_gemm_source;
        cl_program  p = try_build_program_from_source(context, device, src.c_str());
        cl_kernel & k = s == &be_gemm_n64 ? be_k.gemm_n64 : be_k.gemm_n32;
        k = nullptr;
        if (p != nullptr) {
            CL_CHECK((k = clCreateKernel(p, "kernel_gemm_f32", &err), err));
            CL_CHECK(clReleaseProgram(p)); // the kernel keeps the program alive
        }
    }
    for (int i = 0; i < 3; ++i) {
        const std::string src = "#define NC " + std::to_string(be_mmvn_nc[i]) + "\n#define R " + std::to_string(be_mmvn_r) +
                                "\n#define LWG " + std::to_string(be_mmvn_lwg) + "\n" + be_q6r_common + be_mmvn_source;
        cl_program p = try_build_program_from_source(context, device, src.c_str());
        be_k.mmvn_pack[i] = be_k.mmvn_q4_K[i] = be_k.mmvn_q5_K[i] = be_k.mmvn_q6_K[i] = be_k.mmvn_q6r[i] = nullptr;
        if (p != nullptr) {
            CL_CHECK((be_k.mmvn_pack[i] = clCreateKernel(p, "kernel_mmvn_pack", &err), err));
            CL_CHECK((be_k.mmvn_q4_K[i] = clCreateKernel(p, "kernel_mmvn_q4_K", &err), err));
            CL_CHECK((be_k.mmvn_q5_K[i] = clCreateKernel(p, "kernel_mmvn_q5_K", &err), err));
            CL_CHECK((be_k.mmvn_q6_K[i] = clCreateKernel(p, "kernel_mmvn_q6_K", &err), err));
            CL_CHECK((be_k.mmvn_q6r[i]  = clCreateKernel(p, "kernel_mmvn_q6r",  &err), err));
            CL_CHECK(clReleaseProgram(p)); // the kernels keep the program alive
        }
    }
    {
        be_dmmvn_max_cols = (int) std::min<size_t>(ggml_cl_env_size("GGML_OPENCL_DMMVN_MAX_COLS", (size_t) be_dmmvn_max_cols), 4);
        for (int i = 0; i < 3; ++i) {
            be_k.dmmvn_q4_K[i] = be_k.dmmvn_q5_K[i] = be_k.dmmvn_q6r[i] = be_k.dmmvn_q6r_v2[i] = nullptr;
            if (2 + i > be_dmmvn_max_cols) {
                continue;
            }
            const std::string src = "#define NC " + std::to_string(2 + i) + "\n" + be_q6r_common + be_ncol_common + be_dmmvn_source;
            cl_program p = try_build_program_from_source(context, device, src.c_str());
            if (p != nullptr) {
                CL_CHECK((be_k.dmmvn_q4_K[i]   = clCreateKernel(p, "kernel_dmmvn_q4_K",   &err), err));
                CL_CHECK((be_k.dmmvn_q5_K[i]   = clCreateKernel(p, "kernel_dmmvn_q5_K",   &err), err));
                CL_CHECK((be_k.dmmvn_q6r[i]    = clCreateKernel(p, "kernel_dmmvn_q6r",    &err), err));
                CL_CHECK((be_k.dmmvn_q6r_v2[i] = clCreateKernel(p, "kernel_dmmvn_q6r_v2", &err), err));
                CL_CHECK(clReleaseProgram(p)); // the kernels keep the program alive
            }
        }
    }

    // Q8_0 in the q8r layout (be_q8_source), one program per NC = 1..4 (GGML_OPENCL_Q8R=0: Q8_0 on the CPU)
    {
        const char * q8r_env = getenv("GGML_OPENCL_Q8R");
        const bool   q8r_off = q8r_env != nullptr && strcmp(q8r_env, "0") == 0;
        be_k.get_rows_q8r = be_k.dequant_q8r = nullptr;
        for (int i = 0; i < 4; ++i) {
            be_k.dmmv_q8r[i] = be_k.dmmv_q8r_v2[i] = nullptr;
            if (q8r_off) {
                continue;
            }
            const std::string src = "#define NC " + std::to_string(1 + i) + "\n" + be_ncol_common + be_q8_source;
            cl_program p = try_build_program_from_source(context, device, src.c_str());
            if (p != nullptr) {
                CL_CHECK((be_k.dmmv_q8r[i]    = clCreateKernel(p, "kernel_dmmv_q8r",    &err), err));
                CL_CHECK((be_k.dmmv_q8r_v2[i] = clCreateKernel(p, "kernel_dmmv_q8r_v2", &err), err));
                if (i == 0) {
                    CL_CHECK((be_k.get_rows_q8r = clCreateKernel(p, "kernel_get_rows_q8r", &err), err));
                    CL_CHECK((be_k.dequant_q8r  = clCreateKernel(p, "kernel_dequant_q8r",  &err), err));
                }
                CL_CHECK(clReleaseProgram(p)); // the kernels keep the program alive
            }
        }
    }

    // limits of the device and the paths that fit them; the paths that do not fit fall back to slower
    // ones that work on any OpenCL 1.1 device (GGML_OPENCL_TEST_* lower the limits to test them)
    be_dq_max_size = (size_t) std::min<cl_ulong>(256u*1024u*1024u, ggml_cl_max_alloc());
    be_gemm_n64_ok = ggml_cl_kernel_fits(be_k.gemm_n64, (size_t) (be_gemm_n64.bm / be_gemm_n64.tm) * (be_gemm_n64.bn / be_gemm_n64.tn), 0);
    be_gemm_n32_ok = ggml_cl_kernel_fits(be_k.gemm_n32, (size_t) (be_gemm_n32.bm / be_gemm_n32.tm) * (be_gemm_n32.bn / be_gemm_n32.tn), 0);
    be_tiled_ok    = ggml_cl_kernel_fits(be_k.mul_mat_f_tiled, 16*16, 0);
    for (int t = 0; t < 3; ++t) {
        be_mmvn_ok[t] = true;
        for (int i = 0; i < 3; ++i) {
            cl_kernel km = t == 0 ? be_k.mmvn_q4_K[i] : t == 1 ? be_k.mmvn_q5_K[i] : be_k.mmvn_q6_K[i];
            be_mmvn_ok[t] = be_mmvn_ok[t] && be_k.mmvn_pack[i] != nullptr && ggml_cl_kernel_fits(km, be_mmvn_lwg, 0);
        }
    }
    be_mmvn_q6r_ok = true;
    for (int i = 0; i < 3; ++i) {
        be_mmvn_q6r_ok = be_mmvn_q6r_ok && be_k.mmvn_pack[i] != nullptr && ggml_cl_kernel_fits(be_k.mmvn_q6r[i], be_mmvn_lwg, 0);
    }
    // q6r layout: work-groups of up to 256 work-items (at least 8: one row) and chunks of y of up to 2048
    // values (at least 256), as large as the device allows; if nothing fits, the standard layout is kept
    {
        const char * q6r_env = getenv("GGML_OPENCL_Q6R");
        be_q6r_on  = false;
        be_q6r_wg  = std::min(ggml_cl_wg(be_k.dmmv_q6r, 256), ggml_cl_wg(be_k.dmmv_q6r_v2, 256));
        be_q6r_ych = 0;
        if (!(q6r_env != nullptr && strcmp(q6r_env, "0") == 0) && be_q6r_wg >= 8 &&
            ggml_cl_wg(be_k.dequant_q6r, 64) >= 8 && ggml_cl_wg(be_k.get_rows_q6r, 64) >= 8) {
            for (int ych = 2048; ych >= 256; ych /= 2) {
                const size_t lmem = sizeof(float) * (be_q6r_wg + ych + ych/32 + ych/8);
                if (ggml_cl_kernel_fits(be_k.dmmv_q6r, be_q6r_wg, lmem) && ggml_cl_kernel_fits(be_k.dmmv_q6r_v2, be_q6r_wg, lmem)) {
                    be_q6r_ych = ych;
                    be_q6r_on  = true;
                    break;
                }
            }
        }
    }
    // Q4_K / Q5_K matrix x vector with y in local memory: the same search (work-groups of up to 256, chunks of
    // up to 2048 values)
    {
        cl_uint cu = 1;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL));
        be_cu = (int) std::max<cl_uint>(cu, 1);
        const char * q45_env = getenv("GGML_OPENCL_Q45_LY");
        be_q45_on  = false;
        be_q45_wg  = 256;
        for (cl_kernel k : { be_k.dmmv_q4_K_ly1, be_k.dmmv_q4_K_lyc, be_k.dmmv_q5_K_ly1, be_k.dmmv_q5_K_lyc }) {
            be_q45_wg = std::min(be_q45_wg, ggml_cl_wg(k, 256));
        }
        be_q45_ych = 0;
        if (!(q45_env != nullptr && strcmp(q45_env, "0") == 0) && be_q45_wg >= 8) {
            for (int ych = 2048; ych >= 256; ych /= 2) {
                const size_t lmem = sizeof(float) * (be_q45_wg + ych + ych/8);
                if (ggml_cl_kernel_fits(be_k.dmmv_q4_K_ly1, be_q45_wg, lmem) && ggml_cl_kernel_fits(be_k.dmmv_q4_K_lyc, be_q45_wg, lmem) &&
                    ggml_cl_kernel_fits(be_k.dmmv_q5_K_ly1, be_q45_wg, lmem) && ggml_cl_kernel_fits(be_k.dmmv_q5_K_lyc, be_q45_wg, lmem)) {
                    be_q45_ych = ych;
                    be_q45_on  = true;
                    break;
                }
            }
        }
    }
    // q8r: work-groups of up to 256 and the largest chunk of y (at most 2048 values, NC*sch <= 4096) that fits for
    // every NC; the layout is used if the one-column kernel, GET_ROWS and the dequantization fit
    {
        be_q8r_on = false;
        for (int i = 0; i < 4; ++i) {
            const int nc = 1 + i;
            be_q8r_wg[i]  = 0;
            be_q8r_sch[i] = 0;
            if (be_k.dmmv_q8r[i] == nullptr) {
                continue;
            }
            const size_t wg = std::min(ggml_cl_wg(be_k.dmmv_q8r[i], 256), ggml_cl_wg(be_k.dmmv_q8r_v2[i], 256));
            for (int sch = 2048; wg >= 8 && sch >= 128; sch /= 2) {
                const size_t lmem = sizeof(float) * (nc*(wg + 1) + nc*sch);
                if (nc*sch <= 4096 && ggml_cl_kernel_fits(be_k.dmmv_q8r[i], wg, lmem) && ggml_cl_kernel_fits(be_k.dmmv_q8r_v2[i], wg, lmem)) {
                    be_q8r_wg[i]  = wg;
                    be_q8r_sch[i] = sch;
                    break;
                }
            }
        }
        be_q8r_on = be_q8r_sch[0] > 0 && be_k.get_rows_q8r != nullptr && ggml_cl_wg(be_k.get_rows_q8r, 64) >= 8 &&
                    ggml_cl_wg(be_k.dequant_q8r, 64) >= 8;
    }
    // matrix x NC vectors (be_dmmvn_source): the largest chunk of y that fits, for every NC
    {
        const int    ny_max = 4096;
        const size_t wg_max = 256;
        for (int i = 0; i < 3; ++i) {
            const int nc = 2 + i;
            be_dmmvn_q45_wg[i]  = 0;
            be_dmmvn_q45_sch[i] = 0;
            be_dmmvn_q6r_wg[i]  = 0;
            be_dmmvn_q6r_sch[i] = 0;
            if (be_q45_on && be_k.dmmvn_q4_K[i] != nullptr) {
                const size_t wg = std::min(be_q45_wg, std::min(ggml_cl_wg(be_k.dmmvn_q4_K[i], wg_max), ggml_cl_wg(be_k.dmmvn_q5_K[i], wg_max)));
                for (int sch = std::min(2048, be_q45_ych); wg >= 8 && sch >= 256; sch /= 2) {
                    const size_t lmem = sizeof(float) * (nc*(wg + 1) + nc*sch + nc*sch/8);
                    if (nc*sch <= ny_max && ggml_cl_kernel_fits(be_k.dmmvn_q4_K[i], wg, lmem) && ggml_cl_kernel_fits(be_k.dmmvn_q5_K[i], wg, lmem)) {
                        be_dmmvn_q45_wg[i]  = wg;
                        be_dmmvn_q45_sch[i] = sch;
                        break;
                    }
                }
            }
            if (be_q6r_on && be_k.dmmvn_q6r[i] != nullptr) {
                const size_t wg = std::min(ggml_cl_wg(be_k.dmmvn_q6r[i], wg_max), ggml_cl_wg(be_k.dmmvn_q6r_v2[i], wg_max));
                for (int sch = 2048; wg >= 8 && sch >= 256; sch /= 2) {
                    const size_t lmem = sizeof(float) * (nc*(wg + 1) + nc*(sch + sch/32) + nc*sch/8);
                    if (nc*sch <= ny_max && ggml_cl_kernel_fits(be_k.dmmvn_q6r[i], wg, lmem) && ggml_cl_kernel_fits(be_k.dmmvn_q6r_v2[i], wg, lmem)) {
                        be_dmmvn_q6r_wg[i]  = wg;
                        be_dmmvn_q6r_sch[i] = sch;
                        break;
                    }
                }
            }
        }
    }
    {
        size_t dmax = 0;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(dmax), &dmax, NULL));
        if (ggml_cl_vram_limit() > 0) {
            fprintf(stderr, "ggml_opencl: simulated video memory of %.2f MiB (GGML_OPENCL_TEST_MAX_VRAM)\n", ggml_cl_vram_limit()/1024.0/1024.0);
        }
        fprintf(stderr, "ggml_opencl: local memory %llu KiB, max work-group %zu, max allocation %llu MiB\n",
                (unsigned long long) (ggml_cl_local_mem() / 1024), std::min(dmax, ggml_cl_env_size("GGML_OPENCL_TEST_MAX_WG", dmax)),
                (unsigned long long) (ggml_cl_max_alloc() / (1024*1024)));
        const char * na = "";
        if (!be_gemm_n64_ok)  { fprintf(stderr, "ggml_opencl: kernel_gemm_f32 128x64 does not fit this device\n"); na = "x"; }
        if (!be_gemm_n32_ok)  { fprintf(stderr, "ggml_opencl: kernel_gemm_f32 256x32 does not fit this device\n"); na = "x"; }
        if (!be_tiled_ok)     { fprintf(stderr, "ggml_opencl: kernel_mul_mat_f_tiled does not fit this device\n"); na = "x"; }
        for (int t = 0; t < 3; ++t) {
            if (!be_mmvn_ok[t]) {
                fprintf(stderr, "ggml_opencl: kernel_mmvn_%s does not fit this device\n", t == 0 ? "q4_K" : t == 1 ? "q5_K" : "q6_K");
                na = "x";
            }
        }
        if (be_q6r_on && !be_mmvn_q6r_ok) {
            fprintf(stderr, "ggml_opencl: kernel_mmvn_q6r does not fit this device\n");
            na = "x";
        }
        if (na[0] != '\0') {
            fprintf(stderr, "ggml_opencl: those paths use slower fallbacks\n");
        }
        if (be_q6r_on) {
            fprintf(stderr, "ggml_opencl: Q6_K weights with ne[0] %% 512 == 0 in the q6r layout (work-groups of %zu, y in chunks of %d)\n",
                    be_q6r_wg, be_q6r_ych);
        } else {
            fprintf(stderr, "ggml_opencl: Q6_K weights in the standard layout\n");
        }
        if (be_q45_on) {
            fprintf(stderr, "ggml_opencl: Q4_K/Q5_K matrix x vector with y in local memory (work-groups of %zu, y in chunks of %d)\n",
                    be_q45_wg, be_q45_ych);
        }
        if (be_q8r_on) {
            fprintf(stderr, "ggml_opencl: Q8_0 weights with ne[0] %% 64 == 0 in the q8r layout (work-groups of %zu, chunks of y for 1/2/3/4 columns: %d/%d/%d/%d)\n",
                    be_q8r_wg[0], be_q8r_sch[0], be_q8r_sch[1], be_q8r_sch[2], be_q8r_sch[3]);
        } else {
            fprintf(stderr, "ggml_opencl: no Q8_0 kernels on this device: the Q8_0 operations run on the CPU\n");
        }
        fprintf(stderr, "ggml_opencl: matrix x 2/3/4 vectors, chunks of y (0: mmvn path): Q4_K/Q5_K %d/%d/%d, q6r %d/%d/%d\n",
                be_dmmvn_q45_sch[0], be_dmmvn_q45_sch[1], be_dmmvn_q45_sch[2],
                be_dmmvn_q6r_sch[0], be_dmmvn_q6r_sch[1], be_dmmvn_q6r_sch[2]);
    }
    {
        const char * gemm = getenv("GGML_OPENCL_GEMM");
        if (gemm != nullptr && strcmp(gemm, "tiled") == 0) {
            be_gemm_kind = BE_GEMM_TILED;
        }
        const char * min_cols = getenv("GGML_OPENCL_GEMM_MIN_COLS");
        if (min_cols != nullptr) {
            be_gemm_min_cols = atoll(min_cols);
        }
        const char * mmvn_max = getenv("GGML_OPENCL_MMVN_MAX_COLS");
        if (mmvn_max != nullptr) {
            be_mmvn_max_cols_q45 = atoll(mmvn_max);
            be_mmvn_max_cols_q6  = atoll(mmvn_max);
        }
    }
    initialized = true;
}

// buffer

struct ggml_backend_opencl_buffer_context {
    ~ggml_backend_opencl_buffer_context() {
        if (buffer) {
            clReleaseMemObject(buffer);
            ggml_cl_vram_give(size);
        }
    }

    cl_mem buffer;
    size_t size; // counted in the simulated video memory (GGML_OPENCL_TEST_MAX_VRAM)
};

// one per device of the registry
struct ggml_backend_opencl_device_context {
    int                      index; // in g_cl_devices
    std::string              name;  // OpenCL0, OpenCL1, ...
    std::string              description;
    ggml_backend_buffer_type buft;
};

static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size);

static bool ggml_backend_buffer_is_opencl(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft != nullptr && buffer->buft->iface.alloc_buffer == ggml_backend_opencl_buffer_type_alloc_buffer;
}

// cl_mem and byte offset of the data of a tensor allocated in an OpenCL buffer
static cl_mem ggml_cl_be_mem(const ggml_tensor * t, cl_ulong * off) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    GGML_ASSERT(ggml_backend_buffer_is_opencl(buf));
    *off = (cl_ulong) ((const char *) t->data - (const char *) cl_ptr_base);
    return ((ggml_backend_opencl_buffer_context *) buf->context)->buffer;
}

// ---- q6r layout (see be_q6r_common) ----
// true if the data of t (or of the tensor it is a view of) are stored in the q6r layout
static bool ggml_cl_q6r(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return be_q6r_on && base->type == GGML_TYPE_Q6_K && base->ne[0] % 512 == 0;
}

// true if the data of t (or of the tensor it is a view of) are stored in the q8r layout (see be_q8_source)
static bool ggml_cl_q8r(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return be_q8r_on && base->type == GGML_TYPE_Q8_0 && base->ne[0] % 64 == 0;
}

// true if the kernels of the repacked layouts (q6r, q8r) can read t: whole rows of its base tensor (always
// true for the tensors in the standard layouts)
static bool ggml_cl_q6r_view_ok(const ggml_tensor * t) {
    if (!ggml_cl_q6r(t) && !ggml_cl_q8r(t)) {
        return true;
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    const size_t rs   = base->nb[1];
    const size_t voff = (size_t) ((const char *) t->data - (const char *) base->data);
    return t->ne[0] == base->ne[0] && voff % rs == 0 && t->nb[1] % rs == 0 && t->nb[2] % rs == 0 && t->nb[3] % rs == 0;
}

// one row of nb block_q6_K super-blocks <-> q6r
static void ggml_cl_q6r_pack_row(const uint8_t * src, uint8_t * dst, int nb) {
    uint8_t * qlp = dst;
    uint8_t * qhp = dst + (size_t) nb * 128;
    uint8_t * scp = dst + (size_t) nb * 192;
    uint8_t * dp  = dst + (size_t) nb * 208;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * b = src + (size_t) i * 210;
        for (int j = 0; j < 8; ++j) {
            const int h = j >> 2;
            const int p = j & 3;
            memcpy(qlp + (size_t) i*128 +      8*j, b + 64*h +      8*p, 8);
            memcpy(qlp + (size_t) i*128 + 64 + 8*j, b + 64*h + 32 + 8*p, 8);
            memcpy(qhp + (size_t) i*64  +      8*j, b + 128 + 32*h + 8*p, 8);
        }
        for (int h = 0; h < 2; ++h) {
            for (int is = 0; is < 2; ++is) {
                for (int g = 0; g < 4; ++g) {
                    scp[(size_t) i*16 + 4*(2*h + is) + g] = b[192 + 8*h + is + 2*g];
                }
            }
        }
        memcpy(dp + 2*i, b + 208, 2);
    }
}

static void ggml_cl_q6r_unpack_row(const uint8_t * src, uint8_t * dst, int nb) {
    const uint8_t * qlp = src;
    const uint8_t * qhp = src + (size_t) nb * 128;
    const uint8_t * scp = src + (size_t) nb * 192;
    const uint8_t * dp  = src + (size_t) nb * 208;
    for (int i = 0; i < nb; ++i) {
        uint8_t * b = dst + (size_t) i * 210;
        for (int j = 0; j < 8; ++j) {
            const int h = j >> 2;
            const int p = j & 3;
            memcpy(b + 64*h +      8*p,  qlp + (size_t) i*128 +      8*j, 8);
            memcpy(b + 64*h + 32 + 8*p,  qlp + (size_t) i*128 + 64 + 8*j, 8);
            memcpy(b + 128 + 32*h + 8*p, qhp + (size_t) i*64  +      8*j, 8);
        }
        for (int h = 0; h < 2; ++h) {
            for (int is = 0; is < 2; ++is) {
                for (int g = 0; g < 4; ++g) {
                    b[192 + 8*h + is + 2*g] = scp[(size_t) i*16 + 4*(2*h + is) + g];
                }
            }
        }
        memcpy(b + 208, dp + 2*i, 2);
    }
}

// one row of nb block_q8_0 (half d, int8 qs[32]) <-> q8r (qs[nb][32], then d[nb])
static void ggml_cl_q8r_pack_row(const uint8_t * src, uint8_t * dst, int nb) {
    for (int i = 0; i < nb; ++i) {
        memcpy(dst + (size_t) i*32,              src + (size_t) i*34 + 2, 32);
        memcpy(dst + (size_t) nb*32 + (size_t) 2*i, src + (size_t) i*34,     2);
    }
}

static void ggml_cl_q8r_unpack_row(const uint8_t * src, uint8_t * dst, int nb) {
    for (int i = 0; i < nb; ++i) {
        memcpy(dst + (size_t) i*34 + 2, src + (size_t) i*32,                 32);
        memcpy(dst + (size_t) i*34,     src + (size_t) nb*32 + (size_t) 2*i, 2);
    }
}

// bytes [offset, offset + size) of the block_q6_K (q6r) or block_q8_0 (q8r) data of tensor t, written from
// data (set) or read into data (get). The rows of its base tensor that the range touches are converted in
// chunks of up to 16 MiB; rows that are only partly written are read and restored first.
static void ggml_cl_q6r_transfer(cl_mem mem, const ggml_tensor * t, void * data, size_t offset, size_t size, bool set) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    const bool   q8    = base->type == GGML_TYPE_Q8_0;
    const size_t rs    = base->nb[1];
    const int    nb    = (int) (base->ne[0] / (q8 ? 32 : 256));
    void (*pack_row)(const uint8_t *, uint8_t *, int)   = q8 ? ggml_cl_q8r_pack_row   : ggml_cl_q6r_pack_row;
    void (*unpack_row)(const uint8_t *, uint8_t *, int) = q8 ? ggml_cl_q8r_unpack_row : ggml_cl_q6r_unpack_row;
    const size_t bmem  = (size_t) ((const char *) base->data - (const char *) cl_ptr_base); // offset of base in mem
    const size_t bpos  = (size_t) ((const char *) t->data - (const char *) base->data) + offset; // range start in base
    const size_t r0    = bpos / rs;
    const size_t r1    = (bpos + size + rs - 1) / rs;
    const size_t chunk = std::max<size_t>(1, (16u*1024u*1024u) / rs);
    std::vector<uint8_t> packed(std::min(chunk, r1 - r0) * rs);
    std::vector<uint8_t> plain(packed.size());
    for (size_t ra = r0; ra < r1; ra += chunk) {
        const size_t rb  = std::min(r1, ra + chunk);
        const size_t lo  = ra * rs;                      // byte range of rows ra..rb-1 in base
        const size_t hi  = rb * rs;
        const size_t clo = std::max(lo, bpos);           // part of it inside the requested range
        const size_t chi = std::min(hi, bpos + size);
        const size_t n   = hi - lo;
        if (!set || clo > lo || chi < hi) {
            CL_CHECK(clEnqueueReadBuffer(be_queue, mem, CL_TRUE, bmem + lo, n, packed.data(), 0, NULL, NULL));
            for (size_t r = 0; r < rb - ra; ++r) {
                unpack_row(packed.data() + r*rs, plain.data() + r*rs, nb);
            }
        }
        if (set) {
            memcpy(plain.data() + (clo - lo), (const uint8_t *) data + (clo - bpos), chi - clo);
            for (size_t r = 0; r < rb - ra; ++r) {
                pack_row(plain.data() + r*rs, packed.data() + r*rs, nb);
            }
            CL_CHECK(clEnqueueWriteBuffer(be_queue, mem, CL_TRUE, bmem + lo, n, packed.data(), 0, NULL, NULL));
        } else {
            memcpy((uint8_t *) data + (clo - bpos), plain.data() + (clo - lo), chi - clo);
        }
    }
}

static void ggml_backend_opencl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_opencl_buffer_get_base(ggml_backend_buffer_t buffer) {
    return cl_ptr_base;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_opencl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
}

// writes size bytes of value at byte offset off of mem: clEnqueueFillBuffer is OpenCL 1.2, so the pattern is
// uploaded in chunks with clEnqueueWriteBuffer (blocking, so the host chunk can be reused)
static void ggml_cl_fill(cl_mem mem, size_t off, size_t size, uint8_t value) {
    const size_t chunk_size = std::min<size_t>(size, 16u*1024u*1024u);
    if (chunk_size == 0) {
        return;
    }
    std::vector<uint8_t> host_chunk(chunk_size, value);
    for (size_t done = 0; done < size; done += chunk_size) {
        const size_t n = std::min(chunk_size, size - done);
        CL_CHECK(clEnqueueWriteBuffer(be_queue, mem, CL_TRUE, off + done, n, host_chunk.data(), 0, NULL, NULL));
    }
}

static void ggml_backend_opencl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    if (ggml_cl_q6r(tensor) || ggml_cl_q8r(tensor)) {
        // repacked layouts: the bytes are set in the standard layout and repacked
        std::vector<uint8_t> host(size, value);
        ggml_cl_q6r_transfer(ctx->buffer, tensor, host.data(), offset, size, true);
        return;
    }
    const size_t toff = (size_t) ((const char *) tensor->data - (const char *) cl_ptr_base);
    ggml_cl_fill(ctx->buffer, toff + offset, size, value);
}

static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    if (ggml_cl_q6r(tensor) || ggml_cl_q8r(tensor)) {
        ggml_cl_q6r_transfer(ctx->buffer, tensor, const_cast<void *>(data), offset, size, true);
        return;
    }
    const size_t toff = (size_t) ((const char *) tensor->data - (const char *) cl_ptr_base);
    CL_CHECK(clEnqueueWriteBuffer(be_queue, ctx->buffer, CL_TRUE, toff + offset, size, data, 0, NULL, NULL));
}

static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    if (ggml_cl_q6r(tensor) || ggml_cl_q8r(tensor)) {
        ggml_cl_q6r_transfer(ctx->buffer, tensor, data, offset, size, false);
        return;
    }
    const size_t toff = (size_t) ((const char *) tensor->data - (const char *) cl_ptr_base);
    CL_CHECK(clEnqueueReadBuffer(be_queue, ctx->buffer, CL_TRUE, toff + offset, size, data, 0, NULL, NULL));
}

static bool ggml_backend_opencl_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    cl_ulong doff;
    cl_mem   dmem = ggml_cl_be_mem(dst, &doff);
    if (ggml_backend_buffer_is_opencl(src_buf)) {
        // a raw copy keeps the layout: only valid if both are in the same repacked layout (q6r or q8r) with the
        // same rows, or neither is (otherwise ggml_backend_tensor_copy falls back to get + set)
        const int src_rl = ggml_cl_q6r(src) ? 1 : ggml_cl_q8r(src) ? 2 : 0;
        const int dst_rl = ggml_cl_q6r(dst) ? 1 : ggml_cl_q8r(dst) ? 2 : 0;
        if (src_rl != dst_rl || (src_rl != 0 && (!ggml_cl_q6r_view_ok(src) || !ggml_cl_q6r_view_ok(dst) || src->ne[0] != dst->ne[0]))) {
            return false;
        }
        cl_ulong soff;
        cl_mem   smem = ggml_cl_be_mem(src, &soff);
        CL_CHECK(clEnqueueCopyBuffer(be_queue, smem, dmem, soff, doff, ggml_nbytes(src), 0, NULL, NULL));
        return true;
    }
    if (src_buf != nullptr && ggml_backend_buffer_is_host(src_buf)) {
        ggml_backend_opencl_buffer_set_tensor(dst_buf, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    ggml_cl_fill(ctx->buffer, 0, buffer->size, value);
}

static void ggml_backend_opencl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_opencl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opencl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opencl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_opencl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_opencl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_opencl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opencl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_opencl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_opencl_buffer_clear,
    /* .reset           = */ ggml_backend_opencl_buffer_reset,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// buffer type (one per device)

static const char * ggml_backend_opencl_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    return ((ggml_backend_opencl_device_context *) buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer_type->context;
    if (!ggml_cl_activate(dev_ctx->index)) {
        return nullptr;
    }

    cl_mem mem = ggml_cl_create_buffer_now(size, __func__);
    if (mem == nullptr) {
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context{mem, std::max<size_t>(size, 1)};

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opencl_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer_type->context;
    cl_uint alignment = 0;
    CL_CHECK(clGetDeviceInfo(g_cl_devices[dev_ctx->index].device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &alignment, NULL));
    return alignment / 8; // bits to bytes
}

static size_t ggml_backend_opencl_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer_type->context;
    cl_ulong max_alloc = 0;
    CL_CHECK(clGetDeviceInfo(g_cl_devices[dev_ctx->index].device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL));
    // GGML_OPENCL_TEST_MAX_ALLOC simulates a device with a smaller maximum allocation
    return (size_t) std::min<cl_ulong>(max_alloc, ggml_cl_env_size("GGML_OPENCL_TEST_MAX_ALLOC", (size_t) max_alloc));
}

static const ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ NULL,
};

// operations

// sequential setter of kernel arguments
struct cl_kargs {
    cl_kernel k;
    cl_uint   i = 0;
    explicit cl_kargs(cl_kernel k) : k(k) {}
    template <typename T> cl_kargs & a(const T & v) {
        CL_CHECK(clSetKernelArg(k, i++, sizeof(T), &v));
        return *this;
    }
    cl_kargs & local(size_t bytes) {
        CL_CHECK(clSetKernelArg(k, i++, bytes, NULL));
        return *this;
    }
};

static void ggml_cl_be_run(cl_kernel k, cl_uint dims, const size_t * global, const size_t * local) {
    if (be_profile_gpu) {
        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(be_queue, k, dims, NULL, global, local, 0, NULL, &ev));
        be_node_events.push_back(ev);
    } else {
        CL_CHECK(clEnqueueNDRangeKernel(be_queue, k, dims, NULL, global, local, 0, NULL, NULL));
    }
    if (g_cl_kernel_flush) {
        CL_CHECK(clFlush(be_queue));
    }
}

static size_t ggml_cl_be_round_up(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

static cl_ulong nbu(const ggml_tensor * t, int i) {
    return (cl_ulong) t->nb[i];
}

static cl_int nei(const ggml_tensor * t, int i) {
    return (cl_int) t->ne[i];
}

static bool ggml_cl_be_is_f32_or_f16(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static void ggml_cl_be_binary(cl_kernel k, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    cl_kargs(k).a(m0).a(off0).a(m1).a(off1).a(md).a(offd).a(nei(src0, 0))
               .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
               .a(nei(src1, 0)).a(nei(src1, 1)).a(nei(src1, 2)).a(nei(src1, 3))
               .a(nbu(src1, 1)).a(nbu(src1, 2)).a(nbu(src1, 3))
               .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3));
    const size_t wg        = ggml_cl_wg(k, 64);
    const cl_int nch       = (cl_int) ((src0->ne[0] + 4*wg - 1) / (4*wg)); // chunks of 4*wg values per row
    CL_CHECK(clSetKernelArg(k, 20, sizeof(cl_int), &nch));
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { wg * (size_t) src0->ne[1] * (size_t) nch, (size_t) src0->ne[2], (size_t) src0->ne[3] };
    ggml_cl_be_run(k, 3, global, local);
}

static void ggml_cl_be_scale(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float s;
    float b;
    memcpy(&s, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) dst->op_params + 1, sizeof(float));
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(be_k.scale).a(m0).a(off0).a(md).a(offd).a((cl_float) s).a((cl_float) b).a(n);
    const size_t local  = ggml_cl_wg(be_k.scale, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.scale, 1, &global, &local);
}

static void ggml_cl_be_silu(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(be_k.silu).a(m0).a(off0).a(md).a(offd).a(n);
    const size_t local  = ggml_cl_wg(be_k.silu, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.silu, 1, &global, &local);
}

// unary F32 op with one work-item per element (SILU, SIGMOID, SOFTPLUS)
static void ggml_cl_be_unary(cl_kernel k, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(k).a(m0).a(off0).a(md).a(offd).a(n);
    const size_t local  = ggml_cl_wg(k, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(k, 1, &global, &local);
}

static void ggml_cl_be_concat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int dim = ((const int32_t *) dst->op_params)[0];
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(be_k.concat).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                         .a(nei(dst, 0)).a(nei(dst, 1)).a(nei(dst, 2))
                         .a(nbu(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                         .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2)).a(nbu(src1, 3))
                         .a(nbu(dst, 0)).a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                         .a((cl_int) dim).a(nei(src0, dim)).a(n);
    const size_t local  = ggml_cl_wg(be_k.concat, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.concat, 1, &global, &local);
}

static void ggml_cl_be_ssm_conv1d(ggml_tensor * dst) {
    const ggml_tensor * sx = dst->src[0];
    const ggml_tensor * c  = dst->src[1];
    cl_ulong offx, offc, offd;
    cl_mem mx = ggml_cl_be_mem(sx,  &offx);
    cl_mem mc = ggml_cl_be_mem(c,   &offc);
    cl_mem md = ggml_cl_be_mem(dst, &offd);
    cl_kargs(be_k.ssm_conv1d).a(mx).a(offx).a(nbu(sx, 1)).a(nbu(sx, 2))
                             .a(mc).a(offc).a(nbu(c, 1))
                             .a(md).a(offd).a(nbu(dst, 1)).a(nbu(dst, 2))
                             .a(nei(c, 0)).a(nei(dst, 0));
    // batches: kernel_ssm_conv1d_tiled (16 x 16 work-items), if it fits the device
    const int    d_conv = (int) c->ne[0];
    const size_t xt     = (size_t) 16 * (16 + d_conv - 1) * sizeof(float);
    if (dst->ne[1] >= 16 && ggml_cl_kernel_fits(be_k.ssm_conv1d_tiled, 256, xt)) {
        cl_kargs(be_k.ssm_conv1d_tiled).a(mx).a(offx).a(nbu(sx, 1)).a(nbu(sx, 2))
                                       .a(mc).a(offc).a(nbu(c, 1))
                                       .a(md).a(offd).a(nbu(dst, 1)).a(nbu(dst, 2))
                                       .a((cl_int) d_conv).a(nei(dst, 0)).a(nei(dst, 1)).local(xt);
        const size_t local[3]  = { 16, 16, 1 };
        const size_t global[3] = { ggml_cl_be_round_up((size_t) dst->ne[0], 16), ggml_cl_be_round_up((size_t) dst->ne[1], 16), (size_t) dst->ne[2] };
        ggml_cl_be_run(be_k.ssm_conv1d_tiled, 3, global, local);
        return;
    }
    const size_t wg        = ggml_cl_wg(be_k.ssm_conv1d, 64);
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { ggml_cl_be_round_up((size_t) dst->ne[0], wg), (size_t) dst->ne[1], (size_t) dst->ne[2] };
    ggml_cl_be_run(be_k.ssm_conv1d, 3, global, local);
}

// work-group size of kernel_gated_delta_net_f32: 16 work-items per column of the state, up to 256
static size_t ggml_cl_be_gdn_wg(void) {
    return ggml_cl_wg(be_k.gated_delta_net, 256);
}

static void ggml_cl_be_argmax(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const size_t wg = ggml_cl_wg(be_k.argmax, 256); // power of two (tree reduction)
    cl_kargs(be_k.argmax).a(m0).a(off0).a(nbu(src0, 1)).a(nei(src0, 0)).a(md).a(offd)
                         .local(wg * sizeof(float)).local(wg * sizeof(cl_int));
    const size_t global = wg * (size_t) src0->ne[1];
    ggml_cl_be_run(be_k.argmax, 1, &global, &wg);
}

// st_dst: tensor that receives the state snapshots instead of dst (after o), or nullptr
// rows: the fused GET_ROWS that gathers the state (src[5]) from the recurrent state tensor, or nullptr
static void ggml_cl_be_gated_delta_net(ggml_tensor * dst, const ggml_tensor * st_dst, const ggml_tensor * rows) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * k  = dst->src[1];
    const ggml_tensor * v  = dst->src[2];
    const ggml_tensor * g  = dst->src[3];
    const ggml_tensor * b  = dst->src[4];
    const ggml_tensor * st = rows != nullptr ? rows->src[0] : dst->src[5];
    cl_ulong offq, offk, offv, offg, offb, offs, offd;
    cl_mem mq = ggml_cl_be_mem(q,   &offq);
    cl_mem mk = ggml_cl_be_mem(k,   &offk);
    cl_mem mv = ggml_cl_be_mem(v,   &offv);
    cl_mem mg = ggml_cl_be_mem(g,   &offg);
    cl_mem mb = ggml_cl_be_mem(b,   &offb);
    cl_mem ms = ggml_cl_be_mem(st,  &offs);
    cl_mem md = ggml_cl_be_mem(dst, &offd);
    cl_ulong offr = 0;
    cl_mem   mr   = ms;
    if (rows != nullptr) {
        mr = ggml_cl_be_mem(rows->src[1], &offr);
    }
    const int64_t H   = v->ne[1];
    const int64_t n_t = v->ne[2];
    const int64_t n_s = v->ne[3];
    const size_t  lsz = ggml_cl_be_gdn_wg();
    cl_ulong offso = offd + (cl_ulong) (128 * H * n_t * n_s) * sizeof(float);
    cl_mem   mso   = md;
    if (st_dst != nullptr) {
        mso = ggml_cl_be_mem(st_dst, &offso);
    }
    const cl_int   n_snap      = ((const int32_t *) dst->op_params)[0];
    const cl_ulong snap_stride = (cl_ulong) 128 * 128 * H * n_s; // floats between snapshots
    cl_kargs(be_k.gated_delta_net)
        .a(mq).a(offq).a(nbu(q, 1)).a(nbu(q, 2)).a(nbu(q, 3)).a(nei(q, 1)).a((cl_int) (n_s / q->ne[3]))
        .a(mk).a(offk).a(nbu(k, 1)).a(nbu(k, 2)).a(nbu(k, 3)).a(nei(k, 1)).a((cl_int) (n_s / k->ne[3]))
        .a(mv).a(offv).a(nbu(v, 1)).a(nbu(v, 2)).a(nbu(v, 3))
        .a(mg).a(offg).a(mb).a(offb).a(ms).a(offs)
        .a(mr).a(offr).a((cl_ulong) (rows != nullptr ? st->nb[1] : 0)).a((cl_int) (rows != nullptr))
        .a(md).a(offd).a(mso).a(offso)
        .a((cl_int) H).a((cl_int) n_t).a(n_snap).a(snap_stride)
        .local(256 * sizeof(float)).local(2 * lsz * sizeof(float));
    const size_t local[3]  = { lsz, 1, 1 };
    const size_t global[3] = { 128 / (lsz / 16) * lsz, (size_t) H, (size_t) n_s };
    ggml_cl_be_run(be_k.gated_delta_net, 3, global, local);
}

static void ggml_cl_be_rms_norm(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const size_t lsize = ggml_cl_wg(be_k.rms_norm, 256); // power of two (tree reduction)
    cl_kargs(be_k.rms_norm).a(m0).a(off0).a(md).a(offd).a(nei(src0, 0))
                           .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                           .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3)).a((cl_float) eps)
                           .local(lsize * sizeof(float));
    const size_t local[3]  = { lsize, 1, 1 };
    const size_t global[3] = { lsize * (size_t) src0->ne[1], (size_t) src0->ne[2], (size_t) src0->ne[3] };
    ggml_cl_be_run(be_k.rms_norm, 3, global, local);
}

// type code of kernel_cpy: 0 F32, 1 F16, 2 I32
static cl_int ggml_cl_be_cpy_type(ggml_type type) {
    return type == GGML_TYPE_F16 ? 1 : type == GGML_TYPE_I32 ? 2 : 0;
}

static void ggml_cl_be_arange(ggml_tensor * dst) {
    const float start = ggml_get_op_params_f32(dst, 0);
    const float step  = ggml_get_op_params_f32(dst, 2);
    cl_ulong offd;
    cl_mem md = ggml_cl_be_mem(dst, &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(be_k.arange).a(md).a(offd).a((cl_float) start).a((cl_float) step).a(n);
    const size_t local  = ggml_cl_wg(be_k.arange, 64);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.arange, 1, &global, &local);
}

static void ggml_cl_be_cpy(const ggml_tensor * src0, ggml_tensor * dst) {
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(src0);
    cl_kargs(be_k.cpy).a(m0).a(off0).a(md).a(offd)
                      .a(nei(src0, 0)).a(nei(src0, 1)).a(nei(src0, 2))
                      .a(nbu(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                      .a(nei(dst, 0)).a(nei(dst, 1)).a(nei(dst, 2))
                      .a(nbu(dst, 0)).a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                      .a(ggml_cl_be_cpy_type(src0->type)).a(ggml_cl_be_cpy_type(dst->type)).a(n);
    const size_t local  = ggml_cl_wg(be_k.cpy, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.cpy, 1, &global, &local);
}

static void ggml_cl_be_get_rows(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    if (ggml_cl_q8r(src0)) {
        const size_t wg        = ggml_cl_wg(be_k.get_rows_q8r, 64);
        const size_t local[3]  = { wg, 1, 1 };
        const size_t global[3] = { wg * (size_t) src1->ne[0], (size_t) src1->ne[1], (size_t) src1->ne[2] };
        cl_kargs(be_k.get_rows_q8r).a(m0).a(off0).a(m1).a(off1).a(md).a(offd).a((cl_int) (src0->ne[0] / 32))
                                   .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                                   .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2))
                                   .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3));
        ggml_cl_be_run(be_k.get_rows_q8r, 3, global, local);
        return;
    }
    const bool quant = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K;
    cl_kernel  k     = !quant ? be_k.get_rows : src0->type == GGML_TYPE_Q4_K ? be_k.get_rows_q4_K : src0->type == GGML_TYPE_Q5_K ? be_k.get_rows_q5_K :
                       ggml_cl_q6r(src0) ? be_k.get_rows_q6r : be_k.get_rows_q6_K;
    const size_t wg        = ggml_cl_wg(k, 64);
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { wg * (size_t) src1->ne[0], (size_t) src1->ne[1], (size_t) src1->ne[2] };
    GGML_ASSERT(!quant || wg >= 8); // 8 work-items per super-block
    if (quant) {
        cl_kargs(k).a(m0).a(off0).a(m1).a(off1).a(md).a(offd).a((cl_int) (src0->ne[0] / ggml_blck_size(src0->type)))
                   .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                   .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2))
                   .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3));
        ggml_cl_be_run(k, 3, global, local);
        return;
    }
    // F32/F16 rows: chunks of 4096 values (one work-group each), at most 65535 work-groups in dimension 0
    const size_t wgf  = ggml_cl_wg(be_k.get_rows, 256);
    int64_t      nch  = std::max<int64_t>(1, (src0->ne[0] + 4095) / 4096);
    nch = std::max<int64_t>(1, std::min<int64_t>(nch, 65535 / std::max<int64_t>(1, src1->ne[0])));
    const int64_t csz = (src0->ne[0] + nch - 1) / nch;
    cl_kargs(be_k.get_rows).a(m0).a(off0).a(m1).a(off1).a(md).a(offd).a(nei(src0, 0))
                           .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                           .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2))
                           .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                           .a((cl_int) (src0->type == GGML_TYPE_F16)).a((cl_int) nch).a((cl_int) csz);
    const size_t localf[3]  = { wgf, 1, 1 };
    const size_t globalf[3] = { wgf * (size_t) (src1->ne[0] * nch), (size_t) src1->ne[1], (size_t) src1->ne[2] };
    ggml_cl_be_run(be_k.get_rows, 3, globalf, localf);
}

static void ggml_cl_be_soft_max(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    const uint32_t n_head      = (uint32_t) src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float    m0f         = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1f         = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    cl_ulong off0, offm = 0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    cl_mem mm = src1 ? ggml_cl_be_mem(src1, &offm) : m0;
    const size_t lsize = ggml_cl_wg(be_k.soft_max, 256); // power of two (tree reductions)
    // the mask is broadcast over the heads (dimension 2) and the sequences (dimension 3)
    cl_kargs(be_k.soft_max).a(m0).a(off0).a(mm).a(offm).a(md).a(offd)
                           .a(nei(src0, 0)).a(nei(src0, 1)).a(nei(src0, 2)).a(nbu(src0, 1))
                           .a((cl_ulong) (src1 ? src1->nb[1] : 0)).a((cl_ulong) (src1 ? src1->nb[2] : 0)).a((cl_ulong) (src1 ? src1->nb[3] : 0))
                           .a((cl_int) (src1 ? src1->ne[2] : 1)).a((cl_int) (src1 ? src1->ne[3] : 1))
                           .a(nbu(dst, 1))
                           .a((cl_int) (src1 != nullptr)).a((cl_int) (src1 && src1->type == GGML_TYPE_F16))
                           .a((cl_float) scale).a((cl_float) max_bias).a((cl_float) m0f).a((cl_float) m1f)
                           .a((cl_int) n_head_log2).local(lsize * sizeof(float));
    const size_t local  = lsize;
    const size_t global = lsize * (size_t) ggml_nrows(src0);
    ggml_cl_be_run(be_k.soft_max, 1, &global, &local);
}

static void ggml_cl_be_rope(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int   sections[4];
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(sections,     (int32_t *) dst->op_params + 11, sizeof(int)*4);
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(dst), corr_dims);
    const int kmode = mode == GGML_ROPE_TYPE_NEOX ? 1 : mode == GGML_ROPE_TYPE_MROPE ? 2 : mode == GGML_ROPE_TYPE_IMROPE ? 3 : 0;

    cl_ulong off0, offp, offf = 0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem mp = ggml_cl_be_mem(src1, &offp);
    cl_mem mf = src2 ? ggml_cl_be_mem(src2, &offf) : m0;
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    cl_kargs(be_k.rope).a(m0).a(off0).a(mp).a(offp).a(mf).a(offf).a(md).a(offd)
                       .a(nei(src0, 0)).a(nei(src0, 1)).a(nei(src0, 2))
                       .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                       .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                       .a((cl_int) n_dims).a((cl_int) kmode).a((cl_int) (src2 != nullptr)).a((cl_int) n_offs)
                       .a((cl_int) sections[0]).a((cl_int) sections[1]).a((cl_int) sections[2]).a((cl_int) sections[3])
                       .a((cl_float) theta_scale).a((cl_float) freq_scale).a((cl_float) ext_factor).a((cl_float) attn_factor)
                       .a((cl_float) corr_dims[0]).a((cl_float) corr_dims[1]);
    const size_t wg        = ggml_cl_wg(be_k.rope, 32);
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { ggml_cl_be_round_up((size_t) src0->ne[0] / 2, wg), (size_t) src0->ne[1], (size_t) (src0->ne[2] * src0->ne[3]) };
    ggml_cl_be_run(be_k.rope, 3, global, local);
}

static void ggml_cl_be_set_rows(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (ggml_nelements(src0) == 0) {
        return;
    }
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    if (dst->type == GGML_TYPE_Q8_0) {
        // one work-item per block; in q8r the rows of dst are whole rows of its base tensor (ggml_cl_q6r_view_ok)
        const cl_ulong n  = (cl_ulong) (ggml_nelements(src0) / 32);
        const cl_int   nq = ggml_cl_q8r(dst) ? (cl_int) (dst->view_src ? dst->view_src : dst)->ne[0] : 0;
        cl_kargs(be_k.set_rows_q8_0).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                                    .a(nei(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                                    .a(nei(src1, 1)).a(nei(src1, 2))
                                    .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2))
                                    .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                                    .a((cl_int) (src1->type == GGML_TYPE_I64))
                                    .a(nei(src0, 1)).a(nei(src0, 2)).a(n).a(nq);
        const size_t wg     = ggml_cl_wg(be_k.set_rows_q8_0, 64);
        const size_t global = ggml_cl_be_round_up((size_t) n, wg);
        ggml_cl_be_run(be_k.set_rows_q8_0, 1, &global, &wg);
        return;
    }
    cl_kargs(be_k.set_rows).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                           .a(nei(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                           .a(nei(src1, 1)).a(nei(src1, 2))
                           .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2))
                           .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                           .a((cl_int) (src1->type == GGML_TYPE_I64)).a((cl_int) (dst->type == GGML_TYPE_F16))
                           .a(nei(src0, 1)).a(nei(src0, 2)).a((cl_ulong) ggml_nelements(src0));
    const size_t wg     = ggml_cl_wg(be_k.set_rows, 256);
    const size_t global = ggml_cl_be_round_up((size_t) ggml_nelements(src0), wg);
    ggml_cl_be_run(be_k.set_rows, 1, &global, &wg);
}

static void ggml_cl_be_l2_norm(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const size_t wg = ggml_cl_wg(be_k.l2_norm, 256); // power of two (tree reduction)
    cl_kargs(be_k.l2_norm).a(m0).a(off0).a(md).a(offd)
                          .a(nei(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                          .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3)).a((cl_float) eps).local(wg * sizeof(float));
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { wg * (size_t) src0->ne[1], (size_t) src0->ne[2], (size_t) src0->ne[3] };
    ggml_cl_be_run(be_k.l2_norm, 3, global, local);
}

static void ggml_cl_be_pad(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_uint n = (cl_uint) ggml_nelements(dst);
    cl_kargs(be_k.pad).a(m0).a(off0).a(md).a(offd)
                      .a(nei(dst, 0)).a(nei(dst, 1)).a(nei(dst, 2)).a(nei(dst, 3))
                      .a(nbu(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                      .a((cl_int) ggml_get_op_params_i32(dst, 0)).a((cl_int) ggml_get_op_params_i32(dst, 1))
                      .a((cl_int) ggml_get_op_params_i32(dst, 2)).a((cl_int) ggml_get_op_params_i32(dst, 3))
                      .a((cl_int) ggml_get_op_params_i32(dst, 4)).a((cl_int) ggml_get_op_params_i32(dst, 5))
                      .a((cl_int) ggml_get_op_params_i32(dst, 6)).a((cl_int) ggml_get_op_params_i32(dst, 7)).a(n);
    const size_t local  = ggml_cl_wg(be_k.pad, 256);
    const size_t global = ggml_cl_be_round_up(n, local);
    ggml_cl_be_run(be_k.pad, 1, &global, &local);
}

// ---- FLASH_ATTN_EXT ----
// kernel_flash_attn_f16 is compiled at its first use for every (DK, DV, BQ, GH, split); main is nullptr if it does
// not build or does not fit the device (then the operation is not supported and runs on the CPU).
// GGML_OPENCL_FA=0 disables it. The split variant (few query tokens, see SPLIT in be_fattn_source) also has
// kernel_flash_attn_combine; GGML_OPENCL_FA_SPLIT=0 disables it.
static const int be_fa_wg          = 128;
static const int be_fa_split_tiles = 2;  // tiles of 64 K/V rows per chunk of the split variant (fixed: exactness)
// multiply-adds per launch at most: a long operation (long prompt over a long context) is split in launches of
// consecutive query blocks. Some drivers stop work that runs for seconds: i915 (Intel HD 4000 with Beignet)
// resets the GPU ("GPU HANG ... stopped heartbeat" after 2.5 s), which also spoils the commands queued after it.
// The rows of a block do not depend on the launch, so the results do not change.
static const int64_t be_fa_launch_macs = (int64_t) 1 << 30;

struct ggml_cl_fa_kernels {
    cl_kernel main    = nullptr;
    cl_kernel combine = nullptr;
};
static std::map<std::tuple<int, int, int, int, bool, bool, bool>, ggml_cl_fa_kernels> be_fa_kernels;

static cl_mem be_fa_part_buf  = nullptr; // partial results of the chunks of the split variant
static size_t be_fa_part_size = 0;

// Decomposed FLASH_ATTN_EXT (prompts): with be_fa_dec_min() query tokens or more over be_fa_dec_min_kv K/V rows or more
// it is computed with the tuned kernels of the attention without flash attention: S = K^T Q (kernel_mul_mat_f_tiled),
// P = softmax(S*scale + slope*mask) (kernel_soft_max, in place) and O = V P (kernel_mul_mat_f_tiled over V^T, which
// kernel_transpose_f16 builds once per operation), for chunks of the query tokens so that S stays small
// (be_fa_dec_s_max bytes, at least 16 tokens). kernel_flash_attn_f16 reads all the K/V rows again for every 8 rows
// (2 tokens x 4 heads of Qwen3.5) and ran 3 times slower than the attention without flash attention (prompts 11-19 %
// slower with 2048-4096 tokens of context). With fewer K/V rows (the start of a conversation: the cache is padded to
// 256 rows) most of them are masked, and kernel_flash_attn_f16 skips their reads: on the GT 720M, Qwen3.5-0.8B read a
// prompt of 16 tokens 1.3% faster with it, and one of 512 (512 K/V rows) 3.6% slower. Batches of fewer tokens
// (generation, and the verification of the drafts, which compares a token alone and in a batch) keep
// kernel_flash_attn_f16. GGML_OPENCL_FA_DEC=0 disables it; GGML_OPENCL_FA_DEC_MIN sets the number of tokens (at
// least 16).
static cl_mem        be_fa_dec_buf    = nullptr; // V^T, then S
static size_t        be_fa_dec_size   = 0;
static const size_t  be_fa_dec_s_max  = (size_t) 16 << 20;
static const int64_t be_fa_dec_min_kv = 512;

// the temporary buffer *buf (of *cur bytes) with at least size bytes; false if it does not fit the device, and
// a size that did not fit (*failed) is not tried again
static bool ggml_cl_be_tmp_reserve(cl_mem * buf, size_t * cur, size_t * failed, size_t size, const char * what) {
    if (size <= *cur) {
        return true;
    }
    if (*failed != 0 && size >= *failed) {
        return false;
    }
    if (*buf) {
        CL_CHECK(clReleaseMemObject(*buf));
        ggml_cl_vram_give(*cur);
        *buf = nullptr;
        *cur = 0;
    }
    *buf = ggml_cl_create_buffer_now(size, what);
    if (*buf == nullptr) {
        *failed = size;
        return false;
    }
    *cur = size;
    return true;
}

// layout of a Q8_0 K/V view of flash attention (b0, bh, nq of be_q8kv_common): b0 = -1 in the standard layout. In
// q8r its rows (dimension 1) and dimension 3 must be whole rows of the base tensor, and dimension 2 (the heads)
// whole rows or blocks of the same row. False if the view does not fit.
struct ggml_cl_q8_kv {
    cl_int b0 = -1;
    cl_int bh = 0;
    cl_int nq = 0;
};

static bool ggml_cl_q8_kv_layout(const ggml_tensor * t, ggml_cl_q8_kv * l) {
    *l = ggml_cl_q8_kv();
    if (t->type != GGML_TYPE_Q8_0 || !ggml_cl_q8r(t)) {
        return true;
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    const size_t rs   = base->nb[1];
    const size_t blk  = ggml_type_size(GGML_TYPE_Q8_0);
    const size_t voff = (size_t) ((const char *) t->data - (const char *) base->data);
    if (t->nb[1] % rs != 0 || (t->ne[3] > 1 && t->nb[3] % rs != 0) || (voff % rs) % blk != 0) {
        return false;
    }
    l->b0 = (cl_int) ((voff % rs) / blk);
    l->nq = (cl_int) base->ne[0];
    if (t->ne[2] > 1 && t->nb[2] % rs != 0) {
        if (t->nb[2] % blk != 0) {
            return false;
        }
        l->bh = (cl_int) (t->nb[2] / blk);
    }
    // the last head must end in its base row
    return (int64_t) (l->b0 + (t->ne[2] - 1)*l->bh)*32 + t->ne[0] <= base->ne[0];
}

static ggml_cl_fa_kernels ggml_cl_fa_kernel(int dk, int dv, int bq, int gh, bool split, bool kq8, bool vq8) {
    const auto key = std::make_tuple(dk, dv, bq, gh, split, kq8, vq8);
    const auto it  = be_fa_kernels.find(key);
    if (it != be_fa_kernels.end()) {
        return it->second;
    }
    std::string src = "#define DK " + std::to_string(dk) + "\n#define DV " + std::to_string(dv) +
                      "\n#define BQ " + std::to_string(bq) + "\n#define GH " + std::to_string(gh) +
                      "\n#define WG " + std::to_string(be_fa_wg) + "\n";
    if (split) {
        src += "#define SPLIT " + std::to_string(be_fa_split_tiles) + "\n";
    }
    if (kq8) {
        src += "#define K_Q8\n";
    }
    if (vq8) {
        src += "#define V_Q8\n";
    }
    // GGML_OPENCL_FA_SKIP=0: masked tiles are computed instead of skipped (same result: they scale by 1 and add 0)
    static const bool noskip = getenv("GGML_OPENCL_FA_SKIP") != nullptr && strcmp(getenv("GGML_OPENCL_FA_SKIP"), "0") == 0;
    if (noskip) {
        src += "#define NOSKIP\n";
    }
    src += be_q8kv_common;
    src += be_fattn_source;
    ggml_cl_fa_kernels ks;
    cl_program p = try_build_program_from_source(context, device, src.c_str());
    if (p != nullptr) {
        cl_int err;
        ks.main = clCreateKernel(p, "kernel_flash_attn_f16", &err);
        if (err != CL_SUCCESS) {
            ks.main = nullptr;
        }
        if (split) {
            ks.combine = clCreateKernel(p, "kernel_flash_attn_combine", &err);
            if (err != CL_SUCCESS) {
                ks.combine = nullptr;
            }
        }
        CL_CHECK(clReleaseProgram(p)); // the kernels keep the program alive
    }
    if (ks.main == nullptr || !ggml_cl_kernel_fits(ks.main, be_fa_wg, 0) || (split && ks.combine == nullptr)) {
        if (ks.main != nullptr) {
            CL_CHECK(clReleaseKernel(ks.main));
        }
        if (ks.combine != nullptr) {
            CL_CHECK(clReleaseKernel(ks.combine));
        }
        ks = ggml_cl_fa_kernels();
    }
    be_fa_kernels[key] = ks;
    return ks;
}

// true if the split variant is wanted for these query tokens per work-group: always for 1 (generation and the
// verification batches, which then get the same results); it is used if it builds and its buffer fits
static bool ggml_cl_fa_split(int bq) {
    static const bool on = !(getenv("GGML_OPENCL_FA_SPLIT") != nullptr && strcmp(getenv("GGML_OPENCL_FA_SPLIT"), "0") == 0);
    return on && bq == 1;
}

// the buffer of the partial results of the split variant for size bytes; false if it does not fit the device
// (then the variant without chunks is used)
static bool ggml_cl_be_fa_part_reserve(size_t size) {
    static size_t failed = 0;
    return ggml_cl_be_tmp_reserve(&be_fa_part_buf, &be_fa_part_size, &failed, size, "partial results of flash attention");
}

// query tokens from which FLASH_ATTN_EXT is decomposed (0: never)
static int64_t ggml_cl_fa_dec_min(void) {
    static const int64_t n = getenv("GGML_OPENCL_FA_DEC") != nullptr && strcmp(getenv("GGML_OPENCL_FA_DEC"), "0") == 0 ? 0 :
                             getenv("GGML_OPENCL_FA_DEC_MIN") != nullptr ? std::max(16, atoi(getenv("GGML_OPENCL_FA_DEC_MIN"))) : 16;
    return n;
}

static void ggml_cl_be_run_tiled(cl_mem m0, cl_ulong off0, cl_mem m1, cl_ulong off1, cl_mem md, cl_ulong offd,
                                 cl_int ne00, cl_int ne01, cl_int ne11, cl_int ne12, cl_int ne13,
                                 cl_ulong nb01, cl_ulong nb02, cl_ulong nb03, cl_ulong nb11, cl_ulong nb12, cl_ulong nb13,
                                 cl_ulong nb1, cl_ulong nb2, cl_ulong nb3, cl_int r2, cl_int r3, cl_int src0_f16);

// decomposed FLASH_ATTN_EXT (see be_fa_dec_buf); false if it does not apply or its buffer does not fit the device
static bool ggml_cl_be_flash_attn_dec(ggml_tensor * dst, float scale, float max_bias, float softcap, float m0f, float m1f,
                                      uint32_t n_head_log2) {
    static const bool fits = be_tiled_ok && ggml_cl_kernel_fits(be_k.transpose_f16, 16*16, 0);
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t n_min = ggml_cl_fa_dec_min();
    if (!fits || n_min == 0 || q->ne[1] < n_min || k->ne[1] < be_fa_dec_min_kv || softcap != 0.0f) {
        return false;
    }
    const int64_t N  = q->ne[1];
    const int64_t H  = q->ne[2];
    const int64_t KV = k->ne[1];
    const int64_t DK = k->ne[0];
    const int64_t DV = v->ne[0];
    const int64_t Hk = k->ne[2];
    const int64_t Hv = v->ne[2];
    // query tokens per chunk: multiples of 64 (the tiles of kernel_mul_mat_f_tiled) if S allows it, at least 16
    int64_t nq = (int64_t) (be_fa_dec_s_max / ((size_t) H * KV * sizeof(float)));
    nq = nq >= 64 ? nq / 64 * 64 : std::max<int64_t>(16, nq / 16 * 16);
    nq = std::min(nq, N);
    // the buffer: V^T [KV, DV, Hv] (F16), K [DK, KV, Hk] in F16 if it is Q8_0, S [KV, nq, H]
    const bool   kq8     = k->type == GGML_TYPE_Q8_0;
    const bool   vq8     = v->type == GGML_TYPE_Q8_0;
    const size_t vt_size = ggml_cl_be_round_up((size_t) (Hv * DV * KV) * sizeof(ggml_fp16_t), 256);
    const size_t kf_size = kq8 ? ggml_cl_be_round_up((size_t) (Hk * DK * KV) * sizeof(ggml_fp16_t), 256) : 0;
    const size_t s_off   = vt_size + kf_size;
    const size_t s_size  = (size_t) (H * nq * KV) * sizeof(float);
    static size_t failed = 0;
    if (s_off + s_size > ggml_cl_max_alloc() ||
        !ggml_cl_be_tmp_reserve(&be_fa_dec_buf, &be_fa_dec_size, &failed, s_off + s_size, "decomposed flash attention")) {
        return false;
    }
    ggml_cl_q8_kv kl;
    ggml_cl_q8_kv vl;
    GGML_ASSERT(ggml_cl_q8_kv_layout(k, &kl) && ggml_cl_q8_kv_layout(v, &vl)); // checked by ggml_cl_fa_ok

    cl_ulong offq, offk, offv, offm = 0, offd;
    cl_mem mq = ggml_cl_be_mem(q,   &offq);
    cl_mem mk = ggml_cl_be_mem(k,   &offk);
    cl_mem mv = ggml_cl_be_mem(v,   &offv);
    cl_mem md = ggml_cl_be_mem(dst, &offd);
    cl_mem mm = mask ? ggml_cl_be_mem(mask, &offm) : mq;
    const size_t sm_wg = ggml_cl_wg(be_k.soft_max, 256); // power of two (tree reductions)
    const size_t dq_wg = ggml_cl_wg(be_k.dequant_q8_0_kv, 64);
    for (int64_t iq3 = 0; iq3 < q->ne[3]; ++iq3) {
        const cl_ulong oq = offq + iq3*q->nb[3];
        const cl_ulong ok = offk + (iq3 / (q->ne[3] / k->ne[3]))*k->nb[3];
        const cl_ulong ov = offv + (iq3 / (q->ne[3] / v->ne[3]))*v->nb[3];
        const cl_ulong om = mask ? offm + (iq3 % mask->ne[3])*mask->nb[3] : 0;
        const cl_ulong od = offd + iq3*dst->nb[3];
        // V^T [KV, DV, Hv]
        if (vq8) {
            cl_kargs(be_k.dequant_q8_0_kv).a(mv).a(ov).a(nbu(v, 1)).a(nbu(v, 2)).a(vl.b0).a(vl.bh).a(vl.nq)
                                          .a(be_fa_dec_buf).a((cl_ulong) 0).a((cl_int) DV).a((cl_int) KV).a((cl_int) 1);
            const size_t dl[3] = { dq_wg, 1, 1 };
            const size_t dg[3] = { ggml_cl_be_round_up((size_t) KV, dq_wg), (size_t) DV, (size_t) Hv };
            ggml_cl_be_run(be_k.dequant_q8_0_kv, 3, dg, dl);
        } else {
            cl_kargs(be_k.transpose_f16).a(mv).a(ov).a(nbu(v, 1)).a(nbu(v, 2)).a(be_fa_dec_buf).a((cl_ulong) 0)
                                        .a((cl_int) DV).a((cl_int) KV);
            const size_t tl[3] = { 16, 16, 1 };
            const size_t tg[3] = { ggml_cl_be_round_up((size_t) DV, 16), ggml_cl_be_round_up((size_t) KV, 16), (size_t) Hv };
            ggml_cl_be_run(be_k.transpose_f16, 3, tg, tl);
        }
        // K in F16 [DK, KV, Hk] if it is Q8_0
        cl_mem   mk16  = mk;
        cl_ulong ok16  = ok;
        cl_ulong nbk1  = nbu(k, 1);
        cl_ulong nbk2  = nbu(k, 2);
        if (kq8) {
            cl_kargs(be_k.dequant_q8_0_kv).a(mk).a(ok).a(nbu(k, 1)).a(nbu(k, 2)).a(kl.b0).a(kl.bh).a(kl.nq)
                                          .a(be_fa_dec_buf).a((cl_ulong) vt_size).a((cl_int) DK).a((cl_int) KV).a((cl_int) 0);
            const size_t dl[3] = { dq_wg, 1, 1 };
            const size_t dg[3] = { ggml_cl_be_round_up((size_t) DK, dq_wg), (size_t) KV, (size_t) Hk };
            ggml_cl_be_run(be_k.dequant_q8_0_kv, 3, dg, dl);
            mk16 = be_fa_dec_buf;
            ok16 = vt_size;
            nbk1 = (cl_ulong) (DK*sizeof(ggml_fp16_t));
            nbk2 = (cl_ulong) (DK*KV*sizeof(ggml_fp16_t));
        }
        for (int64_t q0 = 0; q0 < N; q0 += nq) {
            const int64_t nqc = std::min(nq, N - q0);
            // S [KV, nqc, H] = K^T Q
            ggml_cl_be_run_tiled(mk16, ok16, mq, oq + q0*q->nb[1], be_fa_dec_buf, s_off,
                                 (cl_int) DK, (cl_int) KV, (cl_int) nqc, (cl_int) H, 1,
                                 nbk1, nbk2, 0, nbu(q, 1), nbu(q, 2), 0,
                                 (cl_ulong) (KV*sizeof(float)), (cl_ulong) (KV*nqc*sizeof(float)), 0,
                                 (cl_int) (H / Hk), 1, 1);
            // P = softmax(S*scale + slope*mask) in place, with the mask rows of the tokens q0, q0 + 1, ...
            cl_kargs(be_k.soft_max).a(be_fa_dec_buf).a((cl_ulong) s_off).a(mm).a((cl_ulong) (om + (mask ? q0*mask->nb[1] : 0)))
                                   .a(be_fa_dec_buf).a((cl_ulong) s_off)
                                   .a((cl_int) KV).a((cl_int) nqc).a((cl_int) H).a((cl_ulong) (KV*sizeof(float)))
                                   .a((cl_ulong) (mask ? mask->nb[1] : 0)).a((cl_ulong) (mask ? mask->nb[2] : 0)).a((cl_ulong) 0)
                                   .a((cl_int) (mask ? mask->ne[2] : 1)).a((cl_int) 1)
                                   .a((cl_ulong) (KV*sizeof(float)))
                                   .a((cl_int) (mask != nullptr)).a((cl_int) 1)
                                   .a((cl_float) scale).a((cl_float) max_bias).a((cl_float) m0f).a((cl_float) m1f)
                                   .a((cl_int) n_head_log2).local(sm_wg * sizeof(float));
            const size_t sm_global = sm_wg * (size_t) (nqc * H);
            ggml_cl_be_run(be_k.soft_max, 1, &sm_global, &sm_wg);
            // O = V P: element (d, h, token) of dst; its tokens are the columns and its heads the matrices
            ggml_cl_be_run_tiled(be_fa_dec_buf, 0, be_fa_dec_buf, s_off, md, od + q0*dst->nb[2],
                                 (cl_int) KV, (cl_int) DV, (cl_int) nqc, (cl_int) H, 1,
                                 (cl_ulong) (KV*sizeof(ggml_fp16_t)), (cl_ulong) (KV*DV*sizeof(ggml_fp16_t)), 0,
                                 (cl_ulong) (KV*sizeof(float)), (cl_ulong) (KV*nqc*sizeof(float)), 0,
                                 nbu(dst, 2), nbu(dst, 1), 0, (cl_int) (H / Hv), 1, 1);
        }
    }
    return true;
}

// heads per work-group: the largest of 4, 2, 1 that divides the number of query heads per K/V head
static int ggml_cl_fa_gh(const ggml_tensor * q, const ggml_tensor * k) {
    const int64_t rk2 = q->ne[2] / k->ne[2];
    return rk2 % 4 == 0 ? 4 : rk2 % 2 == 0 ? 2 : 1;
}

// query tokens per work-group: 1 for few tokens (generation and the verification of drafts: more work-groups),
// 8/GH for prompts (every K/V row is read once for 8 rows)
static int ggml_cl_fa_bq(const ggml_tensor * q, int gh) {
    return q->ne[1] < 16 ? 1 : 8 / gh;
}

static bool ggml_cl_fa_ok(const ggml_tensor * op) {
    static const bool on = !(getenv("GGML_OPENCL_FA") != nullptr && strcmp(getenv("GGML_OPENCL_FA"), "0") == 0);
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];
    if (!on || op->src[4] != nullptr || op->src[5] != nullptr) {
        return false; // no attention sinks, no per-slice K/V rows
    }
    // K and V in F16 or Q8_0 (the cache with -ctk/-ctv q8_0)
    const bool kq8 = k->type == GGML_TYPE_Q8_0;
    const bool vq8 = v->type == GGML_TYPE_Q8_0;
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_F16 && !kq8) || (v->type != GGML_TYPE_F16 && !vq8) || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (k->ne[0] % 64 != 0 || q->ne[0] != k->ne[0] || v->ne[0] > 4*be_fa_wg || (vq8 && v->ne[0] % 32 != 0) || k->ne[1] != v->ne[1] ||
        q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) {
        return false;
    }
    ggml_cl_q8_kv kl;
    ggml_cl_q8_kv vl;
    if (!ggml_cl_q8_kv_layout(k, &kl) || !ggml_cl_q8_kv_layout(v, &vl)) {
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1])) {
        return false;
    }
    const int gh = ggml_cl_fa_gh(q, k);
    // the K/V head of a work-group must also be the same for v
    if ((q->ne[2] / v->ne[2]) % gh != 0) {
        return false;
    }
    return ggml_cl_fa_kernel((int) k->ne[0], (int) v->ne[0], ggml_cl_fa_bq(q, gh), gh, false, kq8, vq8).main != nullptr;
}

static void ggml_cl_be_flash_attn(ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    const uint32_t n_head      = (uint32_t) q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float    m0f         = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1f         = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    if (ggml_cl_be_flash_attn_dec(dst, scale, max_bias, logit_softcap, m0f, m1f, n_head_log2)) {
        return;
    }

    const int gh = ggml_cl_fa_gh(q, k);
    const int bq = ggml_cl_fa_bq(q, gh);
    const int64_t N  = q->ne[1];
    const int64_t nh = q->ne[2];
    const bool kq8 = k->type == GGML_TYPE_Q8_0;
    const bool vq8 = v->type == GGML_TYPE_Q8_0;
    ggml_cl_q8_kv kl;
    ggml_cl_q8_kv vl;
    GGML_ASSERT(ggml_cl_q8_kv_layout(k, &kl) && ggml_cl_q8_kv_layout(v, &vl)); // checked by ggml_cl_fa_ok
    // split variant: chunks of be_fa_split_tiles tiles of the K/V rows, partial results in be_fa_part_buf
    int  nch   = 1;
    bool split = false;
    if (ggml_cl_fa_split(bq) && ggml_cl_fa_kernel((int) k->ne[0], (int) v->ne[0], bq, gh, true, kq8, vq8).main != nullptr) {
        nch   = (int) ((k->ne[1] + 64*be_fa_split_tiles - 1) / (64*be_fa_split_tiles));
        split = ggml_cl_be_fa_part_reserve((size_t) N * nh * q->ne[3] * nch * (v->ne[0] + 2) * sizeof(float));
        if (!split) {
            nch = 1;
        }
    }
    const ggml_cl_fa_kernels ks = ggml_cl_fa_kernel((int) k->ne[0], (int) v->ne[0], bq, gh, split, kq8, vq8);
    cl_kernel kern = ks.main;
    GGML_ASSERT(kern != nullptr);

    cl_ulong offq, offk, offv, offm = 0, offd;
    cl_mem mq = ggml_cl_be_mem(q,   &offq);
    cl_mem mk = ggml_cl_be_mem(k,   &offk);
    cl_mem mv = ggml_cl_be_mem(v,   &offv);
    cl_mem md = ggml_cl_be_mem(dst, &offd);
    cl_mem mm = mask ? ggml_cl_be_mem(mask, &offm) : mq;
    cl_kargs kargs(kern);
    kargs.a(mq).a(offq).a(nbu(q, 1)).a(nbu(q, 2)).a(nbu(q, 3))
                  .a(mk).a(offk).a(nbu(k, 1)).a(nbu(k, 2)).a(nbu(k, 3))
                  .a(mv).a(offv).a(nbu(v, 1)).a(nbu(v, 2)).a(nbu(v, 3))
                  .a(mm).a(offm).a((cl_ulong) (mask ? mask->nb[1] : 0)).a((cl_ulong) (mask ? mask->nb[2] : 0)).a((cl_ulong) (mask ? mask->nb[3] : 0))
                  .a((cl_int) (mask ? mask->ne[2] : 1)).a((cl_int) (mask ? mask->ne[3] : 1)).a((cl_int) (mask != nullptr))
                  .a(md).a(offd).a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                  .a(nei(q, 1)).a(nei(k, 1))
                  .a((cl_int) (q->ne[2] / k->ne[2])).a((cl_int) (q->ne[3] / k->ne[3]))
                  .a((cl_int) (q->ne[2] / v->ne[2])).a((cl_int) (q->ne[3] / v->ne[3]))
                  .a((cl_float) scale).a((cl_float) logit_softcap).a((cl_float) max_bias).a((cl_float) m0f).a((cl_float) m1f)
                  .a((cl_int) n_head_log2)
                  .a(kl.b0).a(kl.bh).a(kl.nq).a(vl.b0).a(vl.bh).a(vl.nq);
    const cl_uint i_qb0 = kargs.i;
    kargs.a((cl_int) 0);
    if (split) {
        kargs.a((cl_int) nh).a(be_fa_part_buf).a((cl_int) nch);
    }
    // query blocks per launch (see be_fa_launch_macs)
    const int64_t n_qb    = (N + bq - 1) / bq;
    const int64_t qb_macs = (nh / gh) * q->ne[3] * (int64_t) bq * gh * k->ne[1] * (k->ne[0] + v->ne[0]);
    const int64_t qb_step = std::max<int64_t>(1, be_fa_launch_macs / std::max<int64_t>(1, qb_macs));
    for (int64_t qb0 = 0; qb0 < n_qb; qb0 += qb_step) {
        const cl_int qb0_arg = (cl_int) qb0;
        CL_CHECK(clSetKernelArg(kern, i_qb0, sizeof(cl_int), &qb0_arg));
        const size_t local[3]  = { (size_t) be_fa_wg, 1, 1 };
        const size_t global[3] = { (size_t) std::min(qb_step, n_qb - qb0) * nch * be_fa_wg, (size_t) (nh / gh), (size_t) q->ne[3] };
        ggml_cl_be_run(kern, 3, global, local);
    }
    if (split) {
        const size_t cwg = ggml_cl_wg(ks.combine, 64);
        const size_t cgl = cwg * (size_t) (N * nh * q->ne[3]);
        cl_kargs(ks.combine).a(be_fa_part_buf).a((cl_int) nch).a(md).a(offd).a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                            .a((cl_int) N).a((cl_int) nh);
        ggml_cl_be_run(ks.combine, 1, &cgl, &cwg);
    }
}

// RMS_NORM node rn followed by its MUL node mul (fusion): one kernel writes the result of the MUL
static void ggml_cl_be_rms_norm_mul(const ggml_tensor * rn, ggml_tensor * mul) {
    const ggml_tensor * src0 = rn->src[0];
    const ggml_tensor * w    = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    cl_ulong off0, offw, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem mw = ggml_cl_be_mem(w,    &offw);
    cl_mem md = ggml_cl_be_mem(mul,  &offd);
    const size_t wg = ggml_cl_wg(be_k.rms_norm_mul, 256); // as kernel_rms_norm_f32: the same reduction
    cl_kargs(be_k.rms_norm_mul).a(m0).a(off0).a(mw).a(offw).a(md).a(offd)
                               .a(nei(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                               .a(nei(w, 0)).a(nei(w, 1)).a(nei(w, 2)).a(nei(w, 3))
                               .a(nbu(w, 1)).a(nbu(w, 2)).a(nbu(w, 3))
                               .a(nbu(mul, 1)).a(nbu(mul, 2)).a(nbu(mul, 3)).a((cl_float) eps).local(wg * sizeof(float));
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { wg * (size_t) src0->ne[1], (size_t) src0->ne[2], (size_t) src0->ne[3] };
    ggml_cl_be_run(be_k.rms_norm_mul, 3, global, local);
}

// true if nodes i (RMS_NORM) and i + 1 (MUL by a weight) can run as one kernel (GGML_OPENCL_FUSION=0: never)
static bool ggml_cl_be_can_fuse_rms_norm_mul(const ggml_cgraph * graph, int i) {
    static const bool on = !(getenv("GGML_OPENCL_FUSION") != nullptr && strcmp(getenv("GGML_OPENCL_FUSION"), "0") == 0);
    if (!on || !ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return false;
    }
    const ggml_tensor * rn  = graph->nodes[i];
    const ggml_tensor * mul = graph->nodes[i + 1];
    const ggml_tensor * w   = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    return mul->src[0] == rn && w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) && ggml_can_repeat(w, rn) &&
           rn->src[0]->type == GGML_TYPE_F32 && rn->src[0]->nb[0] == sizeof(float) && mul->type == GGML_TYPE_F32 &&
           mul->nb[0] == sizeof(float) && ggml_backend_buffer_is_opencl(w->view_src ? w->view_src->buffer : w->buffer);
}

static void ggml_cl_be_top_k(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl_ulong off0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const size_t wg = ggml_cl_wg(be_k.top_k, 256); // power of two (tree reduction)
    cl_kargs(be_k.top_k).a(m0).a(off0).a(nbu(src0, 1)).a(nei(src0, 0)).a(md).a(offd).a(nbu(dst, 1)).a(nei(dst, 0))
                        .local(wg * sizeof(float)).local(wg * sizeof(cl_int));
    const size_t global = wg * (size_t) ggml_nrows(src0);
    ggml_cl_be_run(be_k.top_k, 1, &global, &wg);
}

// SWIGLU of two tensors (ggml_swiglu_split) or of the two halves of the rows of src0 (ggml_swiglu)
static void ggml_cl_be_swiglu(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int nc = (int) dst->ne[0];
    cl_ulong off0, off1 = 0, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = src1 ? ggml_cl_be_mem(src1, &off1) : m0;
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const ggml_tensor * sb = src1 ? src1 : src0;
    if (src1 == nullptr) {
        const bool swapped = ggml_get_op_params_i32(dst, 1) != 0;
        off1 = off0 + (swapped ? 0 : (cl_ulong) nc * sizeof(float));
        off0 = off0 + (swapped ? (cl_ulong) nc * sizeof(float) : 0);
    }
    cl_kargs(be_k.swiglu).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                         .a((cl_int) nc).a(nei(dst, 2))
                         .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                         .a(nbu(sb, 1)).a(nbu(sb, 2)).a(nbu(sb, 3))
                         .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3));
    const size_t wg        = ggml_cl_wg(be_k.swiglu, 256);
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { ggml_cl_be_round_up((size_t) nc, wg), (size_t) dst->ne[1], (size_t) (dst->ne[2] * dst->ne[3]) };
    ggml_cl_be_run(be_k.swiglu, 3, global, local);
}

static bool ggml_cl_be_is_fast_q(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

// launch of kernel_mul_mat_f_tiled (see its comment); ne13 = number of batches in dimension 3
static void ggml_cl_be_run_tiled(cl_mem m0, cl_ulong off0, cl_mem m1, cl_ulong off1, cl_mem md, cl_ulong offd,
                                 cl_int ne00, cl_int ne01, cl_int ne11, cl_int ne12, cl_int ne13,
                                 cl_ulong nb01, cl_ulong nb02, cl_ulong nb03, cl_ulong nb11, cl_ulong nb12, cl_ulong nb13,
                                 cl_ulong nb1, cl_ulong nb2, cl_ulong nb3, cl_int r2, cl_int r3, cl_int src0_f16) {
    cl_kargs(be_k.mul_mat_f_tiled).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                                  .a(ne00).a(ne01).a(ne11).a(ne12)
                                  .a(nb01).a(nb02).a(nb03).a(nb11).a(nb12).a(nb13)
                                  .a(nb1).a(nb2).a(nb3).a(r2).a(r3).a(src0_f16);
    const size_t local[3]  = { 16, 16, 1 };
    const size_t global[3] = { ggml_cl_be_round_up((size_t) ne01, 64) / 4, ggml_cl_be_round_up((size_t) ne11, 64) / 4, (size_t) ne12 * ne13 };
    ggml_cl_be_run(be_k.mul_mat_f_tiled, 3, global, local);
}

// quantized weights (2D, contiguous) x F32 columns with contiguous dimension 0 (K-quants, and Q8_0 in the q8r
// layout)
static bool ggml_cl_be_mul_mat_q_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return (ggml_cl_be_is_fast_q(src0->type) || ggml_cl_q8r(src0)) && ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) &&
           dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float) && src0->ne[0] % ggml_blck_size(src0->type) == 0 &&
           ggml_cl_q6r_view_ok(src0);
}

// the temporary buffer of the dequantized weights (GEMM path) for size bytes; false if it does not fit the
// device: then the batches use the matrix x vector kernels, slower but without that memory (a model that
// barely fits the video memory still runs). A size that did not fit is not tried again.
static bool ggml_cl_be_dq_reserve(size_t size) {
    static size_t failed_size = 0;
    if (size <= be_dq_size) {
        return true;
    }
    if (failed_size != 0 && size >= failed_size) {
        return false;
    }
    if (be_dq_buf) {
        CL_CHECK(clReleaseMemObject(be_dq_buf));
        ggml_cl_vram_give(be_dq_size);
        be_dq_buf  = nullptr;
        be_dq_size = 0;
    }
    be_dq_buf = ggml_cl_create_buffer_now(size, "dequantized weights of the GEMM");
    if (be_dq_buf == nullptr) {
        failed_size = size;
        fprintf(stderr, "ggml_opencl: no room for %.2f MiB of dequantized weights: the prompts use the slower matrix x vector kernels\n",
                size/1024.0/1024.0);
        return false;
    }
    be_dq_size = size;
    return true;
}

// the buffer of the packed columns of kernel_mmvn for size bytes; false if it does not fit the device: then
// the short batches use the next path. A size that did not fit is not tried again.
static bool ggml_cl_be_yi_reserve(size_t size) {
    static size_t failed_size = 0;
    if (size <= be_yi_size) {
        return true;
    }
    if (failed_size != 0 && size >= failed_size) {
        return false;
    }
    if (be_yi_buf) {
        CL_CHECK(clReleaseMemObject(be_yi_buf));
        ggml_cl_vram_give(be_yi_size);
        be_yi_buf  = nullptr;
        be_yi_size = 0;
    }
    be_yi_buf = ggml_cl_create_buffer_now(size, "packed columns of kernel_mmvn");
    if (be_yi_buf == nullptr) {
        failed_size = size;
        return false;
    }
    be_yi_size = size;
    return true;
}

static void ggml_cl_be_mul_mat_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    GGML_ASSERT(off0 % 4 == 0);
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ncol = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const bool    q6r  = ggml_cl_q6r(src0);
    const bool    q8r  = ggml_cl_q8r(src0); // Q8_0: the GEMM path below, or kernel_dmmv_q8r at the end

    // 2-4 columns (verification batches of multi-token prediction): matrix x NC vectors with the tuned
    // kernels, every column with the result of the matrix x vector kernel (only for 2D src1 and dst)
    if (ncol >= 2 && ncol <= 4 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[1] % sizeof(float) == 0) {
        const int    i   = (int) ncol - 2;
        const cl_int ldy = (cl_int) (src1->nb[1] / sizeof(float));
        const cl_int ldd = (cl_int) (dst->nb[1]  / sizeof(float));
        if (q6r && be_dmmvn_q6r_sch[i] > 0) {
            be_mm_path = "dmmvn-q6r";
            const bool   v2  = src0->nb[1] % 8 == 0 && off0 % 8 == 0;
            cl_kernel    k   = v2 ? be_k.dmmvn_q6r_v2[i] : be_k.dmmvn_q6r[i];
            const size_t wg  = be_dmmvn_q6r_wg[i];
            const int    sch = be_dmmvn_q6r_sch[i];
            const size_t gq  = ggml_cl_be_round_up((size_t) ne01, wg / 8) / (wg / 8) * wg;
            cl_kargs(k).a(m0).a(off0).a(m1).a(off1).a(ldy).a(md).a(offd).a(ldd).a((cl_int) ne00).a((cl_int) ne01).a((cl_int) sch)
                       .local((size_t) ncol * (wg + 1) * sizeof(float)).local((size_t) ncol * (sch + sch/32) * sizeof(float))
                       .local((size_t) ncol * (sch/8) * sizeof(float));
            ggml_cl_be_run(k, 1, &gq, &wg);
            return;
        }
        if (!q6r && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K) && be_dmmvn_q45_sch[i] > 0) {
            be_mm_path = "dmmvn-ly";
            cl_kernel     k   = src0->type == GGML_TYPE_Q4_K ? be_k.dmmvn_q4_K[i] : be_k.dmmvn_q5_K[i];
            const size_t  wg  = be_dmmvn_q45_wg[i];
            const int     sch = be_dmmvn_q45_sch[i];
            const int64_t rpg = (int64_t) (wg / 8);
            // T groups of rows per work-group when y is staged once (as in the matrix x vector path)
            int T = 1;
            if (ne00 <= sch) {
                static const int test_T = (int) ggml_cl_env_size("GGML_OPENCL_TEST_Q45_T", 0);
                if (test_T > 0) {
                    T = test_T;
                } else {
                    for (int t : { 4, 2 }) {
                        if ((ne01 + rpg*t - 1) / (rpg*t) >= 8 * be_cu) {
                            T = t;
                            break;
                        }
                    }
                }
            }
            const size_t gq = (size_t) ((ne01 + rpg*T - 1) / (rpg*T)) * wg;
            cl_kargs(k).a(m0).a(off0).a(m1).a(off1).a(ldy).a(md).a(offd).a(ldd).a((cl_int) ne00).a((cl_int) ne01)
                       .a((cl_int) be_q45_ych).a((cl_int) sch).a((cl_int) T)
                       .local((size_t) ncol * (wg + 1) * sizeof(float)).local((size_t) ncol * sch * sizeof(float))
                       .local((size_t) ncol * (sch/8) * sizeof(float));
            ggml_cl_be_run(k, 1, &gq, &wg);
            return;
        }
    }

    // short batches: matrix x several vectors (only for 2D src1 and dst with contiguous rows, and if
    // its kernels fit the device)
    const int     tidx     = src0->type == GGML_TYPE_Q4_K ? 0 : src0->type == GGML_TYPE_Q5_K ? 1 : 2;
    const int64_t mmvn_max = src0->type == GGML_TYPE_Q6_K ? be_mmvn_max_cols_q6 : be_mmvn_max_cols_q45;
    const bool    mmvn_ok  = q6r ? be_mmvn_q6r_ok : be_mmvn_ok[tidx];
    if (!q8r && mmvn_ok && ncol >= 2 && ncol <= mmvn_max && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[1] % sizeof(float) == 0 && ne01 % be_mmvn_r == 0 && ne00 % ggml_blck_size(src0->type) == 0 &&
        ggml_cl_be_yi_reserve((size_t) ne00 * be_mmvn_nc[2] * sizeof(float))) {
        be_mm_path = "mmvn";
        const cl_int ldy  = (cl_int) (src1->nb[1] / sizeof(float));
        const cl_int ldd  = (cl_int) (dst->nb[1]  / sizeof(float));
        const size_t rows_per_wg = (size_t) (be_mmvn_lwg / 8) * be_mmvn_r;
        const size_t gm          = ggml_cl_be_round_up((size_t) ne01, rows_per_wg) / rows_per_wg * be_mmvn_lwg;
        const size_t lm          = be_mmvn_lwg;
        for (int64_t n0 = 0; n0 < ncol; ) {
            const int64_t rest = ncol - n0;
            const int     v    = rest >= 5 ? 2 : rest >= 3 ? 1 : 0;
            const int     nc   = be_mmvn_nc[v];
            const cl_int  ny   = (cl_int) std::min<int64_t>(rest, nc);
            cl_kernel kp = be_k.mmvn_pack[v];
            cl_kernel km = src0->type == GGML_TYPE_Q4_K ? be_k.mmvn_q4_K[v] : src0->type == GGML_TYPE_Q5_K ? be_k.mmvn_q5_K[v] :
                           q6r ? be_k.mmvn_q6r[v] : be_k.mmvn_q6_K[v];
            cl_kargs(kp).a(m1).a((cl_ulong) (off1 + n0*src1->nb[1])).a(ldy).a(ny).a((cl_int) ne00).a(be_yi_buf);
            const size_t lp = ggml_cl_wg(kp, 256);
            const size_t gp = ggml_cl_be_round_up((size_t) ne00 * nc, lp);
            ggml_cl_be_run(kp, 1, &gp, &lp);
            cl_kargs(km).a(m0).a(off0).a(be_yi_buf).a(md).a((cl_ulong) (offd + n0*dst->nb[1])).a(ldd)
                        .a((cl_int) ne00).a((cl_int) ne01).a(ny);
            ggml_cl_be_run(km, 1, &gm, &lm);
            n0 += ny;
        }
        return;
    }

    // batches: dequantize to F32 and run a GEMM (only for 2D, contiguous src1 and dst). kernel_gemm_f32
    // needs K % BK == 0 and 32-bit indices (see be_gemm_source); the tile shape is the preferred one for
    // the number of columns if it fits the device, else the other one, else the generic tiled kernel
    // (GGML_OPENCL_GEMM=tiled forces it); if no GEMM kernel fits, one matrix x vector product per column
    const size_t  dq_size = (size_t) ne00 * ne01 * sizeof(float);
    const int64_t ldb     = src1->nb[1] / sizeof(float);
    const int64_t ldc     = dst->nb[1]  / sizeof(float);
    const bool    own_ok  = ne00 % be_gemm_n64.bk == 0 && ne00 % be_gemm_n32.bk == 0 &&
                            ne01*ne00 < INT32_MAX && ncol*ldb < INT32_MAX && ncol*ldc < INT32_MAX;
    const bool    narrow  = ncol <= be_gemm_n32.bn;
    const bool    use_n32 = be_gemm_n32_ok && (narrow || !be_gemm_n64_ok);
    const bool    use_n64 = be_gemm_n64_ok && !use_n32;
    const bool    own     = be_gemm_kind == BE_GEMM_OWN && own_ok && (use_n32 || use_n64);
    const bool    gemm_ok = own || be_tiled_ok;
    if (gemm_ok && ncol > be_gemm_min_cols && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        dq_size <= be_dq_max_size && ggml_cl_be_dq_reserve(dq_size)) {
        be_mm_path = "gemm";
        cl_kernel kd = src0->type == GGML_TYPE_Q4_K ? be_k.dequant_q4_K : src0->type == GGML_TYPE_Q5_K ? be_k.dequant_q5_K :
                       q8r ? be_k.dequant_q8r : q6r ? be_k.dequant_q6r : be_k.dequant_q6_K;
        const cl_uint  nblocks = (cl_uint) (ne00 * ne01 / ggml_blck_size(src0->type));
        const cl_ulong zero    = 0;
        cl_kargs dargs(kd);
        if (q8r) {
            // one work-item per 4 weights
            dargs.a(m0).a(off0).a(be_dq_buf).a(zero).a((cl_uint) (ne00 * ne01 / 4)).a((cl_int) (ne00 / 32));
            const size_t local  = ggml_cl_wg(kd, 64);
            const size_t global = ggml_cl_be_round_up((size_t) (ne00 * ne01 / 4), local);
            ggml_cl_be_run(kd, 1, &global, &local);
        } else {
            dargs.a(m0).a(off0).a(be_dq_buf).a(zero).a(nblocks);
            if (q6r) {
                dargs.a((cl_int) (ne00 / ggml_blck_size(src0->type)));
            }
            const size_t local  = ggml_cl_wg(kd, 64);
            const size_t global = ggml_cl_be_round_up((size_t) nblocks * 8, local);
            ggml_cl_be_run(kd, 1, &global, &local);
        }

        if (own) {
            const ggml_cl_be_gemm_shape & s = use_n32 ? be_gemm_n32 : be_gemm_n64;
            cl_kernel kg = use_n32 ? be_k.gemm_n32 : be_k.gemm_n64;
            be_mm_path = use_n32 ? "gemm-n32" : "gemm-n64";
            cl_kargs(kg).a(be_dq_buf).a((cl_int) ne00).a(m1).a(off1).a((cl_int) ldb)
                        .a(md).a(offd).a((cl_int) ldc)
                        .a((cl_int) ne01).a((cl_int) ncol).a((cl_int) ne00);
            const size_t local[2]  = { (size_t) (s.bm / s.tm), (size_t) (s.bn / s.tn) };
            const size_t global[2] = { ggml_cl_be_round_up((size_t) ne01, s.bm) / s.bm * local[0],
                                       ggml_cl_be_round_up((size_t) ncol, s.bn) / s.bn * local[1] };
            ggml_cl_be_run(kg, 2, global, local);
            return;
        }
        // generic tiled kernel (GGML_OPENCL_GEMM=tiled, or when kernel_gemm_f32 cannot be used); gemm_ok
        // guarantees that it fits the device
        be_mm_path = "gemm-tiled";
        const cl_ulong nb01d = (cl_ulong) ne00 * sizeof(float);
        ggml_cl_be_run_tiled(be_dq_buf, 0, m1, off1, md, offd, (cl_int) ne00, (cl_int) ne01, (cl_int) ncol, 1, 1,
                             nb01d, nb01d * ne01, nb01d * ne01, nbu(src1, 1), nbu(src1, 2), nbu(src1, 3),
                             nbu(dst, 1), nbu(dst, 2), nbu(dst, 3), 1, 1, 0);
        return;
    }

    // Q8_0 (q8r): kernel_dmmv_q8r in launches of up to 4 columns (the same result per column for any number
    // of columns: decoding a token or verifying it with drafts gives the same logits)
    if (q8r) {
        be_mm_path = "dmmv-q8r";
        const bool   v2  = src0->nb[1] % 16 == 0 && off0 % 16 == 0;
        const cl_int ldy = (cl_int) (src1->nb[1] / sizeof(float));
        const cl_int ldd = (cl_int) (dst->nb[1]  / sizeof(float));
        GGML_ASSERT(src1->nb[1] % sizeof(float) == 0 && dst->nb[1] % sizeof(float) == 0);
        for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
            for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
                for (int64_t i11 = 0; i11 < src1->ne[1]; ) {
                    int nc = (int) std::min<int64_t>(4, src1->ne[1] - i11);
                    while (nc > 1 && be_q8r_sch[nc - 1] == 0) {
                        --nc;
                    }
                    const int     i   = nc - 1;
                    cl_kernel     k   = v2 ? be_k.dmmv_q8r_v2[i] : be_k.dmmv_q8r[i];
                    const size_t  wg  = be_q8r_wg[i];
                    const int     sch = be_q8r_sch[i];
                    const int64_t rpg = (int64_t) (wg / 8);
                    // T groups of rows per work-group when y is staged once (as in the Q4_K/Q5_K kernels)
                    int T = 1;
                    if (ne00 <= sch) {
                        static const int test_T = (int) ggml_cl_env_size("GGML_OPENCL_TEST_Q45_T", 0);
                        if (test_T > 0) {
                            T = test_T;
                        } else {
                            for (int t : { 4, 2 }) {
                                if ((ne01 + rpg*t - 1) / (rpg*t) >= 8 * be_cu) {
                                    T = t;
                                    break;
                                }
                            }
                        }
                    }
                    const size_t   gq   = (size_t) ((ne01 + rpg*T - 1) / (rpg*T)) * wg;
                    const cl_ulong yoff = off1 + i11*src1->nb[1] + i12*src1->nb[2] + i13*src1->nb[3];
                    const cl_ulong doff = offd + i11*dst->nb[1]  + i12*dst->nb[2]  + i13*dst->nb[3];
                    cl_kargs(k).a(m0).a(off0).a(m1).a(yoff).a(ldy).a(md).a(doff).a(ldd).a((cl_int) ne00).a((cl_int) ne01)
                               .a((cl_int) sch).a((cl_int) T)
                               .local((size_t) nc * (wg + 1) * sizeof(float)).local((size_t) nc * sch * sizeof(float));
                    ggml_cl_be_run(k, 1, &gq, &wg);
                    i11 += nc;
                }
            }
        }
        return;
    }

    // one matrix x vector product per column
    if (q6r) {
        be_mm_path = "dmmv-q6r";
        const bool     v2  = src0->nb[1] % 8 == 0 && off0 % 8 == 0;
        cl_kernel      k   = v2 ? be_k.dmmv_q6r_v2 : be_k.dmmv_q6r;
        const size_t   wg  = be_q6r_wg;
        const int      ych = be_q6r_ych;
        const size_t   gq  = ggml_cl_be_round_up((size_t) ne01, wg / 8) / (wg / 8) * wg;
        for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
            for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
                for (int64_t i11 = 0; i11 < src1->ne[1]; ++i11) {
                    const cl_ulong yoff = off1 + i11*src1->nb[1] + i12*src1->nb[2] + i13*src1->nb[3];
                    const cl_ulong doff = offd + i11*dst->nb[1]  + i12*dst->nb[2]  + i13*dst->nb[3];
                    cl_kargs(k).a(m0).a(off0).a(m1).a(yoff).a(md).a(doff).a((cl_int) ne00).a((cl_int) ne01).a((cl_int) ych)
                               .local(wg * sizeof(float)).local((size_t) (ych + ych/32) * sizeof(float)).local((size_t) (ych/8) * sizeof(float));
                    ggml_cl_be_run(k, 1, &gq, &wg);
                }
            }
        }
        return;
    }
    if (be_q45_on && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K)) {
        be_mm_path = "dmmv-ly";
        const bool    one = ne00 <= be_q45_ych;
        cl_kernel     k   = src0->type == GGML_TYPE_Q4_K ? (one ? be_k.dmmv_q4_K_ly1 : be_k.dmmv_q4_K_lyc) :
                                                           (one ? be_k.dmmv_q5_K_ly1 : be_k.dmmv_q5_K_lyc);
        const size_t  wg  = be_q45_wg;
        const int     ych = one ? (int) ne00 : be_q45_ych;
        // T groups of rows per work-group (only with y in one chunk): the largest of 4, 2, 1 that still
        // leaves at least 8 work-groups per compute unit
        const int64_t rpg = (int64_t) (wg / 8);
        // (GGML_OPENCL_TEST_Q45_T forces T, to test that path on devices with many compute units)
        int T = 1;
        if (one) {
            static const int test_T = (int) ggml_cl_env_size("GGML_OPENCL_TEST_Q45_T", 0);
            if (test_T > 0) {
                T = test_T;
            } else {
                for (int t : { 4, 2 }) {
                    if ((ne01 + rpg*t - 1) / (rpg*t) >= 8 * be_cu) {
                        T = t;
                        break;
                    }
                }
            }
        }
        const size_t gq = (size_t) ((ne01 + rpg*T - 1) / (rpg*T)) * wg;
        for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
            for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
                for (int64_t i11 = 0; i11 < src1->ne[1]; ++i11) {
                    const cl_ulong yoff = off1 + i11*src1->nb[1] + i12*src1->nb[2] + i13*src1->nb[3];
                    const cl_ulong doff = offd + i11*dst->nb[1]  + i12*dst->nb[2]  + i13*dst->nb[3];
                    cl_kargs args(k);
                    args.a(m0).a(off0).a(m1).a(yoff).a(md).a(doff).a((cl_int) ne00).a((cl_int) ne01).a((cl_int) (one ? T : ych))
                        .local(wg * sizeof(float)).local((size_t) ych * sizeof(float)).local((size_t) (ych/8) * sizeof(float));
                    ggml_cl_be_run(k, 1, &gq, &wg);
                }
            }
        }
        return;
    }
    be_mm_path = "dmmv";
    size_t     l8_local   = 0;
    bool       l8_local_y = false;
    cl_kernel * k = ggml_get_dmmv_l8_cl(src0->type, ne00, &l8_local, &l8_local_y);
    GGML_ASSERT(k != nullptr);
    const size_t rows_per_wg = l8_local / 8;
    const size_t global      = ggml_cl_be_round_up((size_t) ne01, rows_per_wg) / rows_per_wg * l8_local;
    for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            for (int64_t i11 = 0; i11 < src1->ne[1]; ++i11) {
                const cl_ulong yoff = off1 + i11*src1->nb[1] + i12*src1->nb[2] + i13*src1->nb[3];
                const cl_ulong doff = offd + i11*dst->nb[1]  + i12*dst->nb[2]  + i13*dst->nb[3];
                cl_kargs args(*k);
                args.a(m0).a(off0).a(m1).a(yoff).a(md).a(doff).a((cl_int) ne00).a((cl_int) ne01).local(l8_local * sizeof(float));
                if (l8_local_y) {
                    args.local((size_t) ne00 * sizeof(float));
                }
                ggml_cl_be_run(*k, 1, &global, &l8_local);
            }
        }
    }
}

// generic F32/F16 x F32 product with any strides and broadcasting (attention)
static bool ggml_cl_be_mul_mat_f_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_cl_be_is_f32_or_f16(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           dst->nb[0] == sizeof(float) && src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0;
}

static void ggml_cl_be_mul_mat_f(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    be_mm_path = "generic";
    cl_ulong off0, off1, offd;
    cl_mem m0 = ggml_cl_be_mem(src0, &off0);
    cl_mem m1 = ggml_cl_be_mem(src1, &off1);
    cl_mem md = ggml_cl_be_mem(dst,  &offd);
    const cl_int r2 = (cl_int) (src1->ne[2] / src0->ne[2]);
    const cl_int r3 = (cl_int) (src1->ne[3] / src0->ne[3]);
    // the Hadamard rotations of a quantized K/V cache (llama_mul_mat_hadamard) have several columns per token, and a
    // token must get the same result alone or in a batch of a few tokens (the verification of the drafts compares
    // them): up to 512 columns they keep kernel_mul_mat_vec_f, which computes every column in the same way
    const bool few = src1->ne[1] < 16 || (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD && src1->ne[1] <= 512);
    // batches (attention while processing the prompt): tiled kernel, if both inputs are contiguous in
    // the dimension of the sum
    if (be_tiled_ok && !few && src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float)) {
        be_mm_path = "tiled";
        ggml_cl_be_run_tiled(m0, off0, m1, off1, md, offd, nei(src0, 0), nei(src0, 1), nei(src1, 1), nei(src1, 2), nei(src1, 3),
                             nbu(src0, 1), nbu(src0, 2), nbu(src0, 3), nbu(src1, 1), nbu(src1, 2), nbu(src1, 3),
                             nbu(dst, 1), nbu(dst, 2), nbu(dst, 3), r2, r3, (cl_int) (src0->type == GGML_TYPE_F16));
        return;
    }
    // a few columns (attention while generating): kernel_mul_mat_vec_f, if the rows of src0 and src1 are
    // contiguous and work-groups of at least 32 fit the device (GGML_OPENCL_MMV=0 disables it)
    static const size_t mmv_wg = getenv("GGML_OPENCL_MMV") != nullptr && strcmp(getenv("GGML_OPENCL_MMV"), "0") == 0 ? 0 :
                                 ggml_cl_kernel_fits(be_k.mul_mat_vec_f, ggml_cl_wg(be_k.mul_mat_vec_f, 256), 4 * ggml_cl_wg(be_k.mul_mat_vec_f, 256) * sizeof(float)) ?
                                 ggml_cl_wg(be_k.mul_mat_vec_f, 256) : 0;
    if (mmv_wg >= 32 && few && src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float)) {
        be_mm_path = "vec";
        const int lpr = src0->ne[0] <= 512 ? 8 : 32; // lanes per row
        cl_kargs(be_k.mul_mat_vec_f).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                                    .a(nei(src0, 0)).a(nei(src0, 1)).a(nei(src1, 1))
                                    .a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                                    .a(nbu(src1, 1)).a(nbu(src1, 2)).a(nbu(src1, 3))
                                    .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                                    .a(r2).a(r3).a((cl_int) (src0->type == GGML_TYPE_F16)).a((cl_int) lpr).local(4 * mmv_wg * sizeof(float));
        const size_t rows      = mmv_wg / lpr;
        const size_t local[3]  = { mmv_wg, 1, 1 };
        const size_t global[3] = { ggml_cl_be_round_up((size_t) src0->ne[1], rows) / rows * mmv_wg, (size_t) src0->ne[2], (size_t) src1->ne[3] };
        ggml_cl_be_run(be_k.mul_mat_vec_f, 3, global, local);
        return;
    }
    cl_kargs(be_k.mul_mat_f).a(m0).a(off0).a(m1).a(off1).a(md).a(offd)
                            .a(nei(src0, 0)).a(nei(src0, 1)).a(nei(src1, 1)).a(nei(src1, 2))
                            .a(nbu(src0, 0)).a(nbu(src0, 1)).a(nbu(src0, 2)).a(nbu(src0, 3))
                            .a(nbu(src1, 0)).a(nbu(src1, 1)).a(nbu(src1, 2)).a(nbu(src1, 3))
                            .a(nbu(dst, 1)).a(nbu(dst, 2)).a(nbu(dst, 3))
                            .a(r2).a(r3).a((cl_int) (src0->type == GGML_TYPE_F16));
    const size_t wg        = ggml_cl_wg(be_k.mul_mat_f, 64);
    const size_t local[3]  = { wg, 1, 1 };
    const size_t global[3] = { ggml_cl_be_round_up((size_t) src0->ne[1], wg), (size_t) src1->ne[1], (size_t) (src1->ne[2] * src1->ne[3]) };
    ggml_cl_be_run(be_k.mul_mat_f, 3, global, local);
}

static void ggml_cl_be_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (ggml_cl_be_mul_mat_q_ok(src0, src1, dst)) {
        ggml_cl_be_mul_mat_q(src0, src1, dst);
    } else {
        ggml_cl_be_mul_mat_f(src0, src1, dst);
    }
}

// backend

static ggml_guid_t ggml_backend_opencl_guid() {
    static ggml_guid guid = { 0x6f, 0x70, 0x65, 0x6e, 0x63, 0x6c, 0x2d, 0x67, 0x74, 0x37, 0x32, 0x30, 0x6d, 0x2d, 0x62, 0x65 };
    return &guid;
}

static const char * ggml_backend_opencl_name(ggml_backend_t backend) {
    return ((ggml_backend_opencl_device_context *) backend->device->context)->name.c_str();
}

static void ggml_cl_be_print_profile(void);

static void ggml_backend_opencl_free(ggml_backend_t backend) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    if (be_profile) {
        ggml_cl_be_print_profile();
    }
    delete backend;
}

static void ggml_backend_opencl_synchronize(ggml_backend_t backend) {
    CL_CHECK(clFinish(be_queue));

    GGML_UNUSED(backend);
}

static void ggml_cl_be_print_profile(void) {
    if (be_prof.empty()) {
        return;
    }
    std::vector<std::pair<std::string, ggml_cl_be_prof_entry>> rows(be_prof.begin(), be_prof.end());
    std::sort(rows.begin(), rows.end(), [](const std::pair<std::string, ggml_cl_be_prof_entry> & a, const std::pair<std::string, ggml_cl_be_prof_entry> & b) { return a.second.ms > b.second.ms; });
    double total = 0.0;
    for (const auto & r : rows) {
        total += r.second.ms;
    }
    fprintf(stderr, "\nggml_opencl profile (GGML_OPENCL_PROFILE): %.1f ms in total\n", total);
    fprintf(stderr, "  %-34s %10s %12s %10s %6s\n", "operation", "calls", "total ms", "ms/call", "%");
    for (const auto & r : rows) {
        fprintf(stderr, "  %-34s %10lld %12.1f %10.3f %5.1f%%\n", r.first.c_str(), (long long) r.second.count,
                r.second.ms, r.second.ms / r.second.count, 100.0 * r.second.ms / total);
    }
    be_prof.clear();
}

static bool ggml_cl_be_overlap(const ggml_tensor * a, const char * b0, size_t bn) {
    const char * a0 = (const char *) a->data;
    return a0 < b0 + bn && b0 < a0 + ggml_nbytes(a);
}

// true if the data of tensors a and b overlap (every buffer has the same base address: cl_ptr_base)
static bool ggml_cl_be_overlap_t(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    const ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;
    return ba == bb && ggml_nbytes(a) > 0 && ggml_cl_be_overlap(a, (const char *) b->data, ggml_nbytes(b));
}

static bool ggml_cl_be_is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// Graph fusion for the recurrent layers of qwen35: the state of the sequence is gathered with a GET_ROWS
// (node i, row rs_s_copy[0] of the recurrent state tensor) and GATED_DELTA_NET reads that copy (src[5]).
// With one sequence the kernel reads the row straight from the state tensor through the index instead:
// one copy of 1 MiB per layer and token less (4 ms per token on the GT 720M with Qwen3.5). It needs one
// index, nothing but that GATED_DELTA_NET reading the result of the GET_ROWS, and no node between them
// writing into the state tensor or into the index (the kernel reads both later). With several sequences
// a work-group could read a row that another one overwrites (the snapshots written in place, see
// ggml_cl_be_gdn_state_cpy), so they keep the GET_ROWS. Returns the index of that GATED_DELTA_NET, or -1.
// GGML_OPENCL_GDN_ROWS=0 disables it.
static int ggml_cl_be_gdn_rows(const ggml_cgraph * graph, int i) {
    static const bool on = !(getenv("GGML_OPENCL_GDN_ROWS") != nullptr && strcmp(getenv("GGML_OPENCL_GDN_ROWS"), "0") == 0);
    const ggml_tensor * gr  = graph->nodes[i];
    const ggml_tensor * src = gr->src[0];
    const ggml_tensor * ids = gr->src[1];
    if (!on || gr->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        ggml_nelements(ids) != 1 || src->nb[0] != sizeof(float) || !ggml_is_contiguous(gr) ||
        (gr->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return -1;
    }
    int d = -1;
    for (int u = i + 1; u < graph->n_nodes; ++u) {
        const ggml_tensor * n = graph->nodes[u];
        if (ggml_cl_be_is_view_op(n->op) || ggml_is_empty(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        const bool upto = d < 0; // u <= d
        if (d < 0 && n->op == GGML_OP_GATED_DELTA_NET) {
            const ggml_tensor * st = n->src[5];
            if ((st == gr || st->view_src == gr) && st->data == gr->data && st->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(st) && ggml_nbytes(st) == ggml_nbytes(gr) && st->ne[3] == 1 && n->src[2]->ne[3] == 1) {
                d = u;
            }
        }
        // a node up to the GATED_DELTA_NET (included) that writes into the index: after its last reader (the
        // GET_ROWS of the last recurrent layer) the graph allocator gives its memory to other tensors. Or a
        // node before the GATED_DELTA_NET that writes into the state tensor.
        if (upto && (ggml_cl_be_overlap_t(n, ids) || (u != d && ggml_cl_be_overlap_t(n, src)))) {
            return -1;
        }
        // another reader of the result of the GET_ROWS (the result itself or a view of it: after its last reader
        // the graph allocator gives its memory to other tensors)
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * x = n->src[s];
            if (x != nullptr && !(u == d && s == 5) && (x == gr || x->view_src == gr)) {
                return -1;
            }
        }
    }
    return d;
}

// Graph fusion for the recurrent layers of qwen35: the graph copies the state snapshots of GATED_DELTA_NET
// node i (the part after o of its result: the final state, and with the multi-token prediction also the
// states after the previous tokens) into the recurrent state tensors with a CPY. If nothing else reads that
// part of the result, and no node between them reads the state that the kernel reads (src[5]) or the
// destination of the CPY, the kernel writes the snapshots there directly and the CPY is skipped: 18
// copies of 1 MiB per token less with Qwen3.5-2B. The destination may overlap the state that the kernel
// reads: every work-group reads its columns of the state before writing them. The CPY must copy exactly
// the snapshots that the kernel writes (min(n_t, n_snap)). rows: the fused GET_ROWS of the state (the kernel
// reads the state tensor of the GET_ROWS instead of src[5]), or nullptr. Returns the index of that CPY, or -1.
// GGML_OPENCL_GDN_INPLACE=0 disables it.
static int ggml_cl_be_gdn_state_cpy(const ggml_cgraph * graph, int i, const ggml_tensor * rows) {
    static const bool on = !(getenv("GGML_OPENCL_GDN_INPLACE") != nullptr && strcmp(getenv("GGML_OPENCL_GDN_INPLACE"), "0") == 0);
    if (!on) {
        return -1;
    }
    const ggml_tensor * gdn    = graph->nodes[i];
    const ggml_tensor * st     = gdn->src[5];
    const ggml_tensor * v      = gdn->src[2];
    const int64_t       n_snap = ((const int32_t *) gdn->op_params)[0];
    const int64_t       n_t    = v->ne[2];
    const size_t soff   = (size_t) (v->ne[0] * v->ne[1] * v->ne[2] * v->ne[3]) * sizeof(float);
    const size_t sbytes = ggml_nbytes(st);                               // one state
    const size_t cbytes = (size_t) std::min(n_t, n_snap) * sbytes;     // the snapshots written by the kernel
    if (st->type != GGML_TYPE_F32 || !ggml_is_contiguous(st) || ggml_nbytes(gdn) != soff + (size_t) n_snap * sbytes) {
        return -1;
    }
    const char * sreg = (const char *) gdn->data + soff; // state part of the result
    for (int c = i + 1; c < graph->n_nodes; ++c) {
        const ggml_tensor * n = graph->nodes[c];
        if (n->op != GGML_OP_CPY) {
            continue;
        }
        const ggml_tensor * sv = n->src[0];
        const ggml_tensor * d  = n->src[1];
        if (sv->view_src != gdn || (const char *) sv->data != sreg || !ggml_is_contiguous(sv) || ggml_nbytes(sv) != cbytes ||
            d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nbytes(d) != cbytes) {
            continue;
        }
        for (int u = i + 1; u < graph->n_nodes; ++u) {
            // the view operations read nothing (their consumers are checked through the views)
            const ggml_op op = graph->nodes[u]->op;
            if (u == c || op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
                op == GGML_OP_TRANSPOSE) {
                continue;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = graph->nodes[u]->src[s];
                if (src == nullptr) {
                    continue;
                }
                // another reader of the state part of the result
                if ((src == gdn || src->view_src == gdn) && src != sv && ggml_cl_be_overlap(src, sreg, (size_t) n_snap * sbytes)) {
                    return -1;
                }
                // a reader, before the CPY, of the state read by the kernel or of the destination (it would
                // see the new states)
                if (u < c && (ggml_cl_be_overlap_t(src, rows != nullptr ? rows->src[0] : st) ||
                              ggml_cl_be_overlap_t(src, d))) {
                    return -1;
                }
            }
        }
        return c;
    }
    return -1;
}

static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    std::vector<std::pair<std::string, std::vector<cl_event>>> graph_events; // GGML_OPENCL_PROFILE=2
    std::vector<int> skip; // nodes already computed by a fusion
    std::vector<std::pair<int, int>> gdn_rows; // GATED_DELTA_NET node, GET_ROWS node fused into it
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];

        // nodes without GGML_TENSOR_FLAG_COMPUTE are not computed: e.g. the branches of ggml_build_forward_select
        // that were not selected (the input of llama.cpp builds the token, embedding and mixed paths; the inputs of
        // the unused ones are not set, so running them would read rows at garbage indices)
        if (ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            std::find(skip.begin(), skip.end(), i) != skip.end()) {
            continue;
        }

        const auto t_start = be_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        be_mm_path = "";

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_ADD:
                ggml_cl_be_binary(be_k.add, node);
                break;
            case GGML_OP_MUL:
                ggml_cl_be_binary(be_k.mul, node);
                break;
            case GGML_OP_SCALE:
                ggml_cl_be_scale(node);
                break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU:
                        ggml_cl_be_silu(node);
                        break;
                    case GGML_UNARY_OP_SIGMOID:
                        ggml_cl_be_unary(be_k.sigmoid, node);
                        break;
                    case GGML_UNARY_OP_SOFTPLUS:
                        ggml_cl_be_unary(be_k.softplus, node);
                        break;
                    default:
                        GGML_ASSERT(false);
                }
                break;
            case GGML_OP_CONCAT:
                ggml_cl_be_concat(node);
                break;
            case GGML_OP_SSM_CONV:
                ggml_cl_be_ssm_conv1d(node);
                break;
            case GGML_OP_GATED_DELTA_NET: {
                const ggml_tensor * rows = nullptr;
                for (const auto & p : gdn_rows) {
                    if (p.first == i) {
                        rows = graph->nodes[p.second];
                    }
                }
                const int c = ggml_cl_be_gdn_state_cpy(graph, i, rows);
                ggml_cl_be_gated_delta_net(node, c >= 0 ? graph->nodes[c]->src[1] : nullptr, rows);
                if (c >= 0) {
                    skip.push_back(c);
                }
                break;
            }
            case GGML_OP_RMS_NORM:
                if (ggml_cl_be_can_fuse_rms_norm_mul(graph, i)) {
                    ggml_cl_be_rms_norm_mul(node, graph->nodes[i + 1]);
                    skip.push_back(i + 1);
                    break;
                }
                ggml_cl_be_rms_norm(node);
                break;
            case GGML_OP_ARGMAX:
                ggml_cl_be_argmax(node);
                break;
            case GGML_OP_CPY:
            case GGML_OP_DUP:
            case GGML_OP_CONT:
                ggml_cl_be_cpy(node->src[0], node);
                break;
            case GGML_OP_GET_ROWS: {
                const int d = ggml_cl_be_gdn_rows(graph, i);
                if (d >= 0) {
                    gdn_rows.emplace_back(d, i); // the GATED_DELTA_NET reads the row itself
                    break;
                }
                ggml_cl_be_get_rows(node);
                break;
            }
            case GGML_OP_SOFT_MAX:
                ggml_cl_be_soft_max(node);
                break;
            case GGML_OP_ROPE:
                ggml_cl_be_rope(node);
                break;
            case GGML_OP_SET_ROWS:
                ggml_cl_be_set_rows(node);
                break;
            case GGML_OP_GLU:
                ggml_cl_be_swiglu(node);
                break;
            case GGML_OP_L2_NORM:
                ggml_cl_be_l2_norm(node);
                break;
            case GGML_OP_ARANGE:
                ggml_cl_be_arange(node);
                break;
            case GGML_OP_PAD:
                ggml_cl_be_pad(node);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                ggml_cl_be_flash_attn(node);
                break;
            case GGML_OP_TOP_K:
                ggml_cl_be_top_k(node);
                break;
            case GGML_OP_MUL_MAT:
                ggml_cl_be_mul_mat(node);
                break;
            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                GGML_ASSERT(false);
        }

        // GGML_OPENCL_SYNC=1 (diagnosis): a kernel that fails is reported by the next blocking call, so wait for every
        // node and say which one failed
        if (be_sync) {
            const cl_int err = clFinish(be_queue);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ggml_opencl: node %d '%s' (%s %s) failed with error %d: ne = [%lld, %lld, %lld, %lld], src0 %s [%lld, %lld, %lld, %lld]\n",
                        i, node->name, ggml_op_desc(node), ggml_type_name(node->type), (int) err,
                        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3],
                        node->src[0] ? ggml_type_name(node->src[0]->type) : "-",
                        node->src[0] ? (long long) node->src[0]->ne[0] : 0LL, node->src[0] ? (long long) node->src[0]->ne[1] : 0LL,
                        node->src[0] ? (long long) node->src[0]->ne[2] : 0LL, node->src[0] ? (long long) node->src[0]->ne[3] : 0LL);
                GGML_ABORT("OpenCL kernel failed");
            }
        }

        if (be_profile && node->op != GGML_OP_NONE && node->op != GGML_OP_RESHAPE && node->op != GGML_OP_VIEW &&
            node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE) {
            std::string key = node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
            if (node->op == GGML_OP_MUL_MAT) {
                key += std::string(" ") + be_mm_path + " " + ggml_type_name(node->src[0]->type) + (node->src[1]->ne[1] > 1 ? " batch" : " vec");
            }
            if (be_profile_gpu) {
                graph_events.emplace_back(key, be_node_events);
                be_node_events.clear();
                continue;
            }
            CL_CHECK(clFinish(be_queue));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
            ggml_cl_be_prof_entry & e = be_prof[key];
            e.ms += ms;
            e.count += 1;
        }
    }

    if (be_profile_gpu && !graph_events.empty()) {
        // GPU time of every kernel, and the idle time of the GPU between the first and the last kernel
        CL_CHECK(clFinish(be_queue));
        cl_ulong first = ~(cl_ulong) 0;
        cl_ulong last  = 0;
        double   busy  = 0.0;
        for (auto & ge : graph_events) {
            double ms = 0.0;
            for (cl_event ev : ge.second) {
                cl_ulong t0 = 0;
                cl_ulong t1 = 0;
                CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
                CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL));
                ms   += (double) (t1 - t0) * 1e-6;
                first = std::min(first, t0);
                last  = std::max(last, t1);
                CL_CHECK(clReleaseEvent(ev));
            }
            busy += ms;
            ggml_cl_be_prof_entry & e = be_prof[ge.first];
            e.ms += ms;
            e.count += 1;
        }
        if (last > first) {
            ggml_cl_be_prof_entry & e = be_prof["(GPU idle between kernels)"];
            e.ms += std::max(0.0, (double) (last - first) * 1e-6 - busy);
            e.count += 1;
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

// operations that the backend can run (ggml_backend_sched sends the rest to the CPU backend). It needs the
// active device: the kernels that exist and their limits depend on it.
static bool ggml_cl_be_supports_op(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_are_same_shape(src0, op) && ggml_can_repeat(src1, src0);
        case GGML_OP_SCALE:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(op);
        case GGML_OP_UNARY:
            return (ggml_get_unary_op(op) == GGML_UNARY_OP_SILU || ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID ||
                    ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS) &&
                   src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(op);
        case GGML_OP_GLU: {
            const ggml_glu_op glu = ggml_get_glu_op(op);
            if (glu != GGML_GLU_OP_SWIGLU) { // kernel_swiglu_f32
                return false;
            }
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) &&
                   op->nb[0] == sizeof(float) && (src1 == nullptr || (src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float)));
        }
        case GGML_OP_CONCAT:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_nelements(op) < INT32_MAX;
        case GGML_OP_SSM_CONV:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_GATED_DELTA_NET: {
            // kernel_gated_delta_net_f32 is written for S_k = S_v = 128 with a scalar gate per head and needs
            // work-groups of at least 16
            const ggml_tensor * v = op->src[2];
            const ggml_tensor * g = op->src[3];
            return src0->ne[0] == 128 && src1->ne[0] == 128 && v->ne[0] == 128 && g->ne[0] == 1 &&
                   ggml_is_contiguous(op) && ggml_cl_be_gdn_wg() >= 16;
        }
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_ARGMAX:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && src0->nb[0] == sizeof(float) && ggml_is_contiguous(op) &&
                   src0->ne[2] == 1 && src0->ne[3] == 1;
        case GGML_OP_PAD:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                   ggml_get_op_params_i32(op, 8) == 0 && ggml_nelements(op) < INT32_MAX;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) < INT32_MAX;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return (ggml_cl_be_is_f32_or_f16(src0->type) || src0->type == GGML_TYPE_I32) &&
                   (ggml_cl_be_is_f32_or_f16(op->type) || op->type == GGML_TYPE_I32) &&
                   ggml_nelements(src0) == ggml_nelements(op) && ggml_nelements(src0) < INT32_MAX;
        case GGML_OP_SET_ROWS:
            // F32, F16 or Q8_0 (the K/V cache with -ctk/-ctv q8_0; in q8r, whole rows of the base tensor)
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
                   (ggml_cl_be_is_f32_or_f16(op->type) || (op->type == GGML_TYPE_Q8_0 && src0->ne[0] % 32 == 0 && ggml_cl_q6r_view_ok(op))) &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == ggml_type_size(op->type) && src1->ne[3] == 1;
        case GGML_OP_GET_ROWS:
            return (ggml_cl_be_is_f32_or_f16(src0->type) || (ggml_cl_be_is_fast_q(src0->type) && src0->ne[0] % ggml_blck_size(src0->type) == 0) ||
                    ggml_cl_q8r(src0)) &&
                   ggml_cl_q6r_view_ok(src0) && src1->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == sizeof(float);
        case GGML_OP_SOFT_MAX:
            // no attention sinks (src[2])
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   op->src[2] == nullptr &&
                   (src1 == nullptr || (ggml_cl_be_is_f32_or_f16(src1->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                                        src1->ne[0] == src0->ne[0] && src1->ne[1] >= src0->ne[1] &&
                                        src0->ne[2] % src1->ne[2] == 0 && src0->ne[3] % src1->ne[3] == 0));
        case GGML_OP_ROPE: {
            const int mode = ((const int32_t *) op->op_params)[2];
            const ggml_tensor * src2 = op->src[2];
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_MROPE ||
                    mode == GGML_ROPE_TYPE_IMROPE) &&
                   (src2 == nullptr || src2->type == GGML_TYPE_F32);
        }
        case GGML_OP_MUL_MAT:
            return ggml_cl_be_mul_mat_q_ok(src0, src1, op) || ggml_cl_be_mul_mat_f_ok(src0, src1, op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cl_fa_ok(op);
        case GGML_OP_TOP_K:
            // k <= 16 (TOPK_MAX of kernel_top_k_f32)
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && src0->nb[0] == sizeof(float) &&
                   op->ne[0] >= 1 && op->ne[0] <= 16 && op->ne[0] <= src0->ne[0] && ggml_is_contiguous(op) &&
                   ggml_is_contiguous(src0) && src0->ne[0] < INT32_MAX;
        default:
            return false;
    }
}

static const ggml_backend_i ggml_backend_opencl_i = {
    /* .get_name                = */ ggml_backend_opencl_name,
    /* .free                    = */ ggml_backend_opencl_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_opencl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opencl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ---- device ----

static const char * ggml_backend_opencl_device_get_name(ggml_backend_dev_t dev) {
    return ((ggml_backend_opencl_device_context *) dev->context)->name.c_str();
}

static const char * ggml_backend_opencl_device_get_description(ggml_backend_dev_t dev) {
    return ((ggml_backend_opencl_device_context *) dev->context)->description.c_str();
}

// OpenCL 1.1 cannot report the free memory of a device: total is its global memory (or the simulated video
// memory of GGML_OPENCL_TEST_MAX_VRAM) and free is total minus the buffers that this process has created on
// it (counted by ggml_cl_vram_take)
static void ggml_backend_opencl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    const int idx = ((ggml_backend_opencl_device_context *) dev->context)->index;
    const size_t limit = ggml_cl_vram_limit();
    *total = limit > 0 ? limit : g_cl_devices[idx].global_mem;
    const size_t used = g_cl_active == idx ? be_vram_used : 0;
    *free = *total > used ? *total - used : 0;
}

static enum ggml_backend_dev_type ggml_backend_opencl_device_get_type(ggml_backend_dev_t dev) {
    const ggml_cl_device_info & info = g_cl_devices[((ggml_backend_opencl_device_context *) dev->context)->index];
    // a GPU that uses the RAM of the computer (e.g. Intel HD Graphics with Beignet) is an integrated GPU: llama.cpp
    // only uses it by default if there is no discrete GPU. An OpenCL implementation on the CPU is only listed
    // when it was selected explicitly (GGML_OPENCL_PLATFORM/DEVICE), and then it is used like a GPU.
    if ((info.type & CL_DEVICE_TYPE_GPU) != 0 && info.unified_memory) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_opencl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opencl_device_get_name(dev);
    props->description = ggml_backend_opencl_device_get_description(dev);
    props->type        = ggml_backend_opencl_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_opencl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_opencl_device_init(ggml_backend_dev_t dev, const char * params) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    if (!ggml_cl_activate(dev_ctx->index)) {
        return nullptr;
    }
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ nullptr,
    };
    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_buffer_type(ggml_backend_dev_t dev) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    return &((ggml_backend_opencl_device_context *) dev->context)->buft;
}

static bool ggml_backend_opencl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    std::lock_guard<std::recursive_mutex> lock(be_mutex);
    // the kernels and their limits are known once the device is active; another active device cannot be used
    if (!ggml_cl_activate(((ggml_backend_opencl_device_context *) dev->context)->index)) {
        return false;
    }
    return ggml_cl_be_supports_op(op);
}

static bool ggml_backend_opencl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->device == dev;
}

// op offload: with part of the model in the RAM (weights in a host buffer), ggml_backend_sched asks whether an
// operation on those weights should run here anyway, copying the weights to the device. It pays off for the
// matrix products of batches (prompt processing): the copy of a weight matrix is amortized over many columns,
// and the GPU computes them much faster than the CPU. GGML_OPENCL_OFFLOAD_MIN_BATCH changes the minimum
// batch (default 32, as the CUDA backend; 0 disables it).
static bool ggml_backend_opencl_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    static const int64_t min_batch = (int64_t) ggml_cl_env_size("GGML_OPENCL_OFFLOAD_MIN_BATCH", 32);
    if (min_batch <= 0 || op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    return op->ne[1] >= min_batch && op->ne[2] == 1 && op->ne[3] == 1;

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_opencl_device_i = {
    /* .get_name             = */ ggml_backend_opencl_device_get_name,
    /* .get_description      = */ ggml_backend_opencl_device_get_description,
    /* .get_memory           = */ ggml_backend_opencl_device_get_memory,
    /* .get_type             = */ ggml_backend_opencl_device_get_type,
    /* .get_props            = */ ggml_backend_opencl_device_get_props,
    /* .init_backend         = */ ggml_backend_opencl_device_init,
    /* .get_buffer_type      = */ ggml_backend_opencl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_opencl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opencl_device_supports_buft,
    /* .offload_op           = */ ggml_backend_opencl_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---- registry ----

static std::vector<std::unique_ptr<ggml_backend_opencl_device_context>> g_cl_dev_ctx;
static std::vector<ggml_backend_device> g_cl_backend_devices;

static const char * ggml_backend_opencl_reg_get_name(ggml_backend_reg_t reg) {
    return "OpenCL";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_opencl_reg_get_device_count(ggml_backend_reg_t reg) {
    return g_cl_backend_devices.size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_opencl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < g_cl_backend_devices.size());
    return &g_cl_backend_devices[index];

    GGML_UNUSED(reg);
}

static const ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    /* .get_name         = */ ggml_backend_opencl_reg_get_name,
    /* .get_device_count = */ ggml_backend_opencl_reg_get_device_count,
    /* .get_device       = */ ggml_backend_opencl_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }
    initialized = true;

    reg = ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_opencl_reg_i,
        /* .context     = */ nullptr,
    };

    ggml_cl_probe_devices();
    g_cl_backend_devices.reserve(g_cl_devices.size());
    for (size_t i = 0; i < g_cl_devices.size(); ++i) {
        auto ctx = std::make_unique<ggml_backend_opencl_device_context>();
        ctx->index       = (int) i;
        ctx->name        = "OpenCL" + std::to_string(i);
        ctx->description = g_cl_devices[i].name + " (" + g_cl_devices[i].platform_name + ", " + g_cl_devices[i].version + ")";
        g_cl_backend_devices.push_back(ggml_backend_device {
            /* .iface   = */ ggml_backend_opencl_device_i,
            /* .reg     = */ &reg,
            /* .context = */ ctx.get(),
        });
        ctx->buft = ggml_backend_buffer_type {
            /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
            /* .device  = */ &g_cl_backend_devices.back(),
            /* .context = */ ctx.get(),
        };
        g_cl_dev_ctx.push_back(std::move(ctx));
    }

    return &reg;
}

ggml_backend_t ggml_backend_opencl_init(void) {
    ggml_backend_reg_t reg = ggml_backend_opencl_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void) {
    ggml_backend_reg_t reg = ggml_backend_opencl_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, 0));
}

bool ggml_backend_is_opencl(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_opencl_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)
