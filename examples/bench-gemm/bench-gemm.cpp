// bench-gemm: micro-benchmark of the matrix x matrix products used while processing a prompt
// (quantized weights x batch of activations), to tune them on the GeForce GT 720M.
//
// Built like bench-dmmv: a unity translation unit that #includes ggml-opencl.cpp, linked against
// ggml-base only (target llama-bench-gemm of OpenCL builds).
//
// For every shape (K = row length of the weights, M = rows of the weights, N = columns of the batch)
// it measures with OpenCL profiling events:
//   - the dequantization of the Q4_K weights to F32 (kernel_dequant_q4_K of the backend),
//   - the backend tiled kernel (kernel_mul_mat_f_tiled) on the F32 weights,
//   - experimental tiled GEMM variants on the F32 weights (tile sizes as #defines),
//   - experimental fused variants that read the Q4_K weights directly,
// and checks every result against a CPU reference on a sample of output positions.
// GFLOPS = 2*M*N*K / time.
//
// Build: cmake --build <build folder> --target llama-bench-gemm (OpenCL builds).
// Environment variables: BENCH_GEMM_BASE=1 (also gemm_f32), BENCH_GEMM_PF=1 (gemm_f32_pf),
// BENCH_GEMM_V4=1 (gemm_f32_v4m0/m1), BENCH_GEMM_FUSED=1 (gemm_q4_K), BENCH_GEMM_DUMP_PTX=<prefix>;
// BENCH_GEMM_MMVN=1 (only the short batches, bench_mmvn), with BENCH_GEMM_REGS=1 (ptxas report of
// kernel_mmvn_*) and BENCH_GEMM_MMVN_PTX=<prefix>.
//
// History of the experiments on a GeForce GT 720M (Fermi GF117, OpenCL 1.1, 2026-11-04), GFLOPS
// with 512 columns (Qwen2.5-1.5B ffn up/down, Qwen3.5-2B ffn up):
//   measured FP32 peak (peak_fma)                                   234
//   backend tiled kernel kernel_mul_mat_f_tiled (generic, 64x64)    60-78
//   (CLBlast, untuned for this device, is not measured here: end to end it is 17% slower than the
//   generic tiled kernel, llama-bench pp512 22.49 against 26.70 t/s)
//   round 1, gemm_f32: 64x64/16/4x4 83-97, 128x64/16/8x4 104-110, 64x128/16/4x8 101-109,
//            96x96/16/6x6 89-96, 128x128/16/8x8 14 (spills), BK 8 worse; fused Q4_K (gemm_q4_K) not
//            better: the dequantization pass is only 3-4% of the time
//   round 2: reqd_work_group_size changes nothing; BK 32 2x slower (24 KB of local memory per
//            work-group); register prefetch (gemm_f32_pf) spills with 256 work-items
//   round 3: float4 local/global accesses (gemm_f32_v4m0/m1) 10-30% slower; coalesced loads matter
//            more than the 2-way bank conflicts of the transposed stores
//   round 4: short batches: 256x32/16/8x4 is the fastest tile with 17-32 columns
//   round 5: the PTX shows that the compiler unrolls neither the k loop nor the load loops;
//            gemm_f32_u (#pragma unroll, fixed load column per work-item, 32-bit indices, clamped
//            rows, PAD 2) 128x64/16/8x4 115-131 (+10-20%); 256x32 with 32 columns 70 -> 81
//   round 6: gemm_f32_u2 (unclamped A rows, B offsets computed once) not better
//   round 7 (short batches, BENCH_GEMM_MMVN=1): matrix x several vectors (kernel_mmvn_*, 1 row per
//            lane group): NC 8 is the best one (8 columns: 4.4-5.8 ms per 8960x1536 Q4_K matrix,
//            against 9.8-11.8 once per column); float4 pointers for y (128-bit loads) 1.6-2.5x slower
//            than vload4, which the compiler splits into scalar loads (1 load per FMA)
//   round 8 (mmvx_q4_K): fma chains + scales decoded per sub-block +13-30%; 2 rows per lane group
//            (R 2) +40-70%; R 4 and NC 16 with R 2 run out of registers. With R 2: Q4_K/Q5_K beat
//            dequantization + GEMM up to ~48 columns, Q6_K up to ~28
// Integrated in the backend: gemm_f32_u as kernel_gemm_f32, 128x64/16/8x4 with more than 32 columns
// and 256x32/16/8x4 with 15-32 columns; be_mmvn_source (R 2, NC 2/4/8, work-groups of 128) for 2-48
// columns (Q4_K/Q5_K) or 2-28 (Q6_K); 1 column: matrix x vector kernel.

#include "ggml-opencl.cpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

// ---------------------------------------------------------------------------------------------
// experimental kernels. C[n][m] = sum_k A[m][k] * B[n][k]; A has M rows of K values (row stride
// lda), B has N rows of K floats (ldb), C has N rows of M floats (ldc). Compile-time parameters:
// BM x BN tile of C per work-group, BK step of k, TM x TN values of C per work-item; the work-group
// is (BM/TM) x (BN/TN). The rows of the local tiles are padded by 1 float (no bank conflicts).
// ---------------------------------------------------------------------------------------------
static const char * gemm_source = MULTILINE_QUOTE(

// like gemm_f32, but the next tile is loaded from global memory into registers while the current one
// is computed (software prefetch); requires BM*BK and BN*BK to be multiples of the work-group size
__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32_pf(__global const float * A, const int lda, __global const float * B, const int ldb,
                 __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float As[BK][BM + 1];
    __local float Bs[BK][BN + 1];

    const int tx   = get_local_id(0);
    const int ty   = get_local_id(1);
    const int nthr = (BM/TM) * (BN/TN);
    const int tid  = ty * (BM/TM) + tx;
    const int m0   = get_group_id(0) * BM;
    const int n0   = get_group_id(1) * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    float ra[(BM*BK)/((BM/TM)*(BN/TN))];
    float rb[(BN*BK)/((BM/TM)*(BN/TN))];
    for (int e = 0; e < (BM*BK)/nthr; ++e) {
        const int li = tid + e*nthr;
        const int m  = m0 + li / BK;
        const int k  = li % BK;
        ra[e] = (m < M && k < K) ? A[(size_t) m*lda + k] : 0.0f;
    }
    for (int e = 0; e < (BN*BK)/nthr; ++e) {
        const int li = tid + e*nthr;
        const int n  = n0 + li / BK;
        const int k  = li % BK;
        rb[e] = (n < N && k < K) ? B[(size_t) n*ldb + k] : 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int e = 0; e < (BM*BK)/nthr; ++e) {
            const int li = tid + e*nthr;
            As[li % BK][li / BK] = ra[e];
        }
        for (int e = 0; e < (BN*BK)/nthr; ++e) {
            const int li = tid + e*nthr;
            Bs[li % BK][li / BK] = rb[e];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int k1 = k0 + BK;
        if (k1 < K) {
            for (int e = 0; e < (BM*BK)/nthr; ++e) {
                const int li = tid + e*nthr;
                const int m  = m0 + li / BK;
                const int k  = k1 + li % BK;
                ra[e] = (m < M && k < K) ? A[(size_t) m*lda + k] : 0.0f;
            }
            for (int e = 0; e < (BN*BK)/nthr; ++e) {
                const int li = tid + e*nthr;
                const int n  = n0 + li / BK;
                const int k  = k1 + li % BK;
                rb[e] = (n < N && k < K) ? B[(size_t) n*ldb + k] : 0.0f;
            }
        }
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM];
            float b[TN];
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][tx + i*(BM/TM)];
            }
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][ty + j*(BN/TN)];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int j = 0; j < TN; ++j) {
        const int n = n0 + ty + j*(BN/TN);
        if (n < N) {
            for (int i = 0; i < TM; ++i) {
                const int m = m0 + tx + i*(BM/TM);
                if (m < M) {
                    C[(size_t) n*ldc + m] = acc[i][j];
                }
            }
        }
    }
}

__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32(__global const float * A, const int lda, __global const float * B, const int ldb,
              __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float As[BK][BM + 1];
    __local float Bs[BK][BN + 1];

    const int tx   = get_local_id(0);
    const int ty   = get_local_id(1);
    const int nthr = (BM/TM) * (BN/TN);
    const int tid  = ty * (BM/TM) + tx;
    const int m0   = get_group_id(0) * BM;
    const int n0   = get_group_id(1) * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int li = tid; li < BM*BK; li += nthr) {
            const int kk = li % BK;
            const int mm = li / BK;
            const int m  = m0 + mm;
            const int k  = k0 + kk;
            As[kk][mm] = (m < M && k < K) ? A[(size_t) m*lda + k] : 0.0f;
        }
        for (int li = tid; li < BN*BK; li += nthr) {
            const int kk = li % BK;
            const int nn = li / BK;
            const int n  = n0 + nn;
            const int k  = k0 + kk;
            Bs[kk][nn] = (n < N && k < K) ? B[(size_t) n*ldb + k] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM];
            float b[TN];
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][tx + i*(BM/TM)];
            }
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][ty + j*(BN/TN)];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int j = 0; j < TN; ++j) {
        const int n = n0 + ty + j*(BN/TN);
        if (n < N) {
            for (int i = 0; i < TM; ++i) {
                const int m = m0 + tx + i*(BM/TM);
                if (m < M) {
                    C[(size_t) n*ldc + m] = acc[i][j];
                }
            }
        }
    }
}

// float4 variant of gemm_f32. The TM (TN) values of a work-item are TM/4 (TN/4) groups of 4
// consecutive rows (columns), so every step of k reads them from local memory as float4 and the
// result is written as float4. The tiles are read from global memory as float4 along k (requires
// K % BK == 0, lda, ldb and ldc multiples of 4 and 16-byte aligned A, B and C). The local rows are
// padded by 4 floats (keeps the float4 alignment). loadmap selects how the work-items share the
// loads: 0 = 4 consecutive work-items read 64 consecutive bytes of one row (2-way bank conflicts when
// they are stored transposed), 1 = 16 consecutive work-items read 16 different rows (no conflicts).
inline void gemm_v4_body(__global const float * A, const int lda, __global const float * B, const int ldb,
                         __global float * C, const int ldc, const int M, const int N, const int K,
                         __local float4 * As4, __local float4 * Bs4, const int loadmap) {
    __local float * As = (__local float *) As4;
    __local float * Bs = (__local float *) Bs4;

    const int tx   = get_local_id(0);
    const int ty   = get_local_id(1);
    const int nthr = (BM/TM) * (BN/TN);
    const int tid  = ty * (BM/TM) + tx;
    const int m0   = get_group_id(0) * BM;
    const int n0   = get_group_id(1) * BN;

    float4 acc[TM/4][TN];
    for (int v = 0; v < TM/4; ++v) {
        for (int j = 0; j < TN; ++j) {
            acc[v][j] = (float4)(0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int li = tid; li < BM*(BK/4); li += nthr) {
            const int r  = loadmap == 0 ? li / (BK/4) : (li % 16) + 16*(li / (4*BK));
            const int k4 = loadmap == 0 ? li % (BK/4) : (li / 16) % (BK/4);
            float4 v = (float4)(0.0f);
            if (m0 + r < M) {
                v = *((__global const float4 *) (A + (size_t) (m0 + r)*lda + k0 + 4*k4));
            }
            As[(4*k4 + 0)*(BM + 4) + r] = v.x;
            As[(4*k4 + 1)*(BM + 4) + r] = v.y;
            As[(4*k4 + 2)*(BM + 4) + r] = v.z;
            As[(4*k4 + 3)*(BM + 4) + r] = v.w;
        }
        for (int li = tid; li < BN*(BK/4); li += nthr) {
            const int r  = loadmap == 0 ? li / (BK/4) : (li % 16) + 16*(li / (4*BK));
            const int k4 = loadmap == 0 ? li % (BK/4) : (li / 16) % (BK/4);
            float4 v = (float4)(0.0f);
            if (n0 + r < N) {
                v = *((__global const float4 *) (B + (size_t) (n0 + r)*ldb + k0 + 4*k4));
            }
            Bs[(4*k4 + 0)*(BN + 4) + r] = v.x;
            Bs[(4*k4 + 1)*(BN + 4) + r] = v.y;
            Bs[(4*k4 + 2)*(BN + 4) + r] = v.z;
            Bs[(4*k4 + 3)*(BN + 4) + r] = v.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < BK; ++kk) {
            float4 a[TM/4];
            float4 b[TN/4];
            for (int v = 0; v < TM/4; ++v) {
                a[v] = As4[kk*((BM + 4)/4) + v*(BM/TM) + tx];
            }
            for (int w = 0; w < TN/4; ++w) {
                b[w] = Bs4[kk*((BN + 4)/4) + w*(BN/TN) + ty];
            }
            for (int v = 0; v < TM/4; ++v) {
                for (int w = 0; w < TN/4; ++w) {
                    acc[v][4*w + 0] += a[v] * b[w].x;
                    acc[v][4*w + 1] += a[v] * b[w].y;
                    acc[v][4*w + 2] += a[v] * b[w].z;
                    acc[v][4*w + 3] += a[v] * b[w].w;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < TN/4; ++w) {
        for (int c = 0; c < 4; ++c) {
            const int n = n0 + w*4*(BN/TN) + 4*ty + c;
            if (n < N) {
                for (int v = 0; v < TM/4; ++v) {
                    const int m = m0 + v*4*(BM/TM) + 4*tx;
                    __global float * d = C + (size_t) n*ldc + m;
                    const float4 r = acc[v][4*w + c];
                    if (m + 3 < M) {
                        *((__global float4 *) d) = r;
                    } else {
                        if (m + 0 < M) { d[0] = r.x; }
                        if (m + 1 < M) { d[1] = r.y; }
                        if (m + 2 < M) { d[2] = r.z; }
                    }
                }
            }
        }
    }
}

__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32_v4m0(__global const float * A, const int lda, __global const float * B, const int ldb,
                   __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float4 As4[BK*((BM + 4)/4)];
    __local float4 Bs4[BK*((BN + 4)/4)];
    gemm_v4_body(A, lda, B, ldb, C, ldc, M, N, K, As4, Bs4, 0);
}

__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32_v4m1(__global const float * A, const int lda, __global const float * B, const int ldb,
                   __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float4 As4[BK*((BM + 4)/4)];
    __local float4 Bs4[BK*((BN + 4)/4)];
    gemm_v4_body(A, lda, B, ldb, C, ldc, M, N, K, As4, Bs4, 1);
}

// same as gemm_f32, but A is read as Q4_K super-blocks (nb per row, 144 bytes each) and dequantized
// while loading the local tile; BK must be 16 (a 16-weight segment never crosses a 32-weight
// sub-block). Every load item handles 4 consecutive weights (one 32-bit word of the quants).
__kernel void gemm_q4_K(__global const uchar * Aq, const int nb, __global const float * B, const int ldb,
                        __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float As[16][BM + 1];
    __local float Bs[16][BN + 1];

    const int tx   = get_local_id(0);
    const int ty   = get_local_id(1);
    const int nthr = (BM/TM) * (BN/TN);
    const int tid  = ty * (BM/TM) + tx;
    const int m0   = get_group_id(0) * BM;
    const int n0   = get_group_id(1) * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        const int ib  = k0 >> 8;          // super-block
        const int kb  = k0 & 255;         // position inside the super-block
        const int p   = kb >> 6;          // 64-weight part
        const int hi  = (kb >> 5) & 1;    // high nibbles?
        const int l0  = kb & 31;          // 0 or 16
        const int is  = 2*p + hi;         // sub-block (scale index)
        for (int li = tid; li < BM*4; li += nthr) {
            const int q  = li & 3;
            const int mm = li >> 2;
            const int m  = m0 + mm;
            float4 v = (float4)(0.0f);
            if (m < M) {
                __global const uchar * blk = Aq + ((size_t) m*nb + ib) * 144;
                const float2 dm = vload_half2(0, (__global const half *) blk);
                __global const uchar * s = blk + 4;
                float sc, mn;
                if (is < 4) {
                    sc = (float) (s[is] & 63);
                    mn = (float) (s[is + 4] & 63);
                } else {
                    sc = (float) ((s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4));
                    mn = (float) ((s[is + 4] >>  4) | ((s[is]     >> 6) << 4));
                }
                const uint w   = ((__global const uint *) (blk + 16))[(p*32 + l0)/4 + q];
                const uint nib = hi ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu);
                v = convert_float4(as_uchar4(nib)) * (dm.x*sc) - dm.y*mn;
            }
            As[4*q + 0][mm] = v.x;
            As[4*q + 1][mm] = v.y;
            As[4*q + 2][mm] = v.z;
            As[4*q + 3][mm] = v.w;
        }
        for (int li = tid; li < BN*16; li += nthr) {
            const int kk = li & 15;
            const int nn = li >> 4;
            const int n  = n0 + nn;
            Bs[kk][nn] = n < N ? B[(size_t) n*ldb + k0 + kk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 16; ++kk) {
            float a[TM];
            float b[TN];
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][tx + i*(BM/TM)];
            }
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][ty + j*(BN/TN)];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int j = 0; j < TN; ++j) {
        const int n = n0 + ty + j*(BN/TN);
        if (n < N) {
            for (int i = 0; i < TM; ++i) {
                const int m = m0 + tx + i*(BM/TM);
                if (m < M) {
                    C[(size_t) n*ldc + m] = acc[i][j];
                }
            }
        }
    }
}

);

// gemm_f32 written for the compiler (round 5): the PTX of gemm_f32 shows that the NVIDIA OpenCL
// compiler unrolls neither the k loop nor the load loops and recomputes 64-bit addresses on every
// iteration. This version (a raw string, so it can use #pragma unroll) unrolls every loop; every
// work-item always loads the same column of k (NTHR % BK == 0) through a pointer that moves by BK;
// the indices are 32-bit; out-of-range rows are clamped instead of branching (their results are never
// stored); the local rows are padded by PAD floats (2: no bank conflicts in the transposed stores).
// Requires K % BK == 0, (BM*BK) % NTHR == 0, (BN*BK) % NTHR == 0 and M*lda, N*ldb < 2^31.
static const char * gemm_u_source = R"CLC(
#define NTHR ((BM/TM) * (BN/TN))
__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32_u(__global const float * A, const int lda, __global const float * B, const int ldb,
                __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float As[BK][BM + PAD];
    __local float Bs[BK][BN + PAD];

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

// gemm_f32_u with fewer live registers (round 6): the rows of A are not clamped (A must have its rows
// padded to a multiple of BM; the extra rows are never stored), so the A loads only need one row step;
// the clamped offsets of the B rows are computed once
__kernel __attribute__((reqd_work_group_size(BM/TM, BN/TN, 1)))
void gemm_f32_u2(__global const float * A, const int lda, __global const float * B, const int ldb,
                 __global float * C, const int ldc, const int M, const int N, const int K) {
    __local float As[BK][BM + PAD];
    __local float Bs[BK][BN + PAD];

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

    int ob[(BN*BK)/NTHR];
    #pragma unroll
    for (int e = 0; e < (BN*BK)/NTHR; ++e) {
        ob[e] = min(n0 + rl + e*(NTHR/BK), N - 1)*ldb + kl;
    }
    const int ra = (NTHR/BK)*lda;
    __global const float * pa = A + (m0 + rl)*lda + kl;
    __global const float * pb = B;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int e = 0; e < (BM*BK)/NTHR; ++e) {
            As[kl][rl + e*(NTHR/BK)] = pa[e*ra];
        }
        #pragma unroll
        for (int e = 0; e < (BN*BK)/NTHR; ++e) {
            Bs[kl][rl + e*(NTHR/BK)] = pb[ob[e]];
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

// FP32 peak: 8 independent fma chains per work-item, 4096 fma per chain
static const char * peak_source = MULTILINE_QUOTE(
__kernel void peak_fma(__global float * out, const float s) {
    float a0 = get_global_id(0) * 1e-9f;
    float a1 = a0 + 0.1f;
    float a2 = a0 + 0.2f;
    float a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f;
    float a5 = a0 + 0.5f;
    float a6 = a0 + 0.6f;
    float a7 = a0 + 0.7f;
    for (int i = 0; i < 1024; ++i) {
        for (int u = 0; u < 4; ++u) {
            a0 = fma(a0, s, 0.5f);
            a1 = fma(a1, s, 0.5f);
            a2 = fma(a2, s, 0.5f);
            a3 = fma(a3, s, 0.5f);
            a4 = fma(a4, s, 0.5f);
            a5 = fma(a5, s, 0.5f);
            a6 = fma(a6, s, 0.5f);
            a7 = fma(a7, s, 0.5f);
        }
    }
    out[get_global_id(0)] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
);

struct gemm_cfg {
    int BM, BN, BK, TM, TN;
    int PAD = 2; // only gemm_f32_u
};

static double event_ms(cl_event ev) {
    cl_ulong t0 = 0;
    cl_ulong t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL));
    return (double) (t1 - t0) * 1e-6;
}

// runs a kernel (arguments already set) iters times after one warm-up run; average ms
static double time_kernel(cl_command_queue pq, cl_kernel k, cl_uint dims, const size_t * global, const size_t * local, int iters) {
    double sum_ms = 0.0;
    for (int it = 0; it < iters + 1; ++it) {
        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(pq, k, dims, NULL, global, local, 0, NULL, &ev));
        CL_CHECK(clWaitForEvents(1, &ev));
        if (it >= 1) {
            sum_ms += event_ms(ev);
        }
        CL_CHECK(clReleaseEvent(ev));
    }
    return sum_ms / iters;
}

struct sample {
    int m, n;
    double ref;
};

static double max_rel_err(cl_command_queue pq, cl_mem dC, int M, const std::vector<sample> & samples) {
    std::vector<float> c((size_t) M);
    double max_ref = 0.0;
    double max_err = 0.0;
    for (const auto & s : samples) {
        float v;
        CL_CHECK(clEnqueueReadBuffer(pq, dC, CL_TRUE, ((size_t) s.n*M + s.m)*sizeof(float), sizeof(float), &v, 0, NULL, NULL));
        max_ref = std::max(max_ref, std::fabs(s.ref));
        max_err = std::max(max_err, std::fabs(v - s.ref));
    }
    return max_ref > 0.0 ? max_err / max_ref : max_err;
}

static cl_program build_with_defines(const gemm_cfg & c) {
    const std::string src = "#define BM " + std::to_string(c.BM) + "\n#define BN " + std::to_string(c.BN) +
                            "\n#define BK " + std::to_string(c.BK) + "\n#define TM " + std::to_string(c.TM) +
                            "\n#define TN " + std::to_string(c.TN) + "\n#define PAD " + std::to_string(c.PAD) + "\n" +
                            gemm_source + gemm_u_source;
    return build_program_from_source(context, device, src.c_str());
}

static void bench_shape(cl_command_queue pq, int K, int M, int N, const char * desc, const std::vector<gemm_cfg> & cfgs) {
    const double flop = 2.0 * M * N * K;
    const int    nb   = K / (int) ggml_blck_size(GGML_TYPE_Q4_K);
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, K);

    // random Q4_K weights (quantized in chunks) and F32 batch
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(row_size * M);
    {
        const int chunk_rows = 256;
        std::vector<float> chunk((size_t) chunk_rows * K);
        for (int r0 = 0; r0 < M; r0 += chunk_rows) {
            const int nr = std::min(chunk_rows, M - r0);
            for (size_t i = 0; i < (size_t) nr * K; ++i) {
                chunk[i] = dist(rng);
            }
            ggml_quantize_chunk(GGML_TYPE_Q4_K, chunk.data(), q.data() + (size_t) r0 * row_size, 0, nr, K, nullptr);
        }
    }
    std::vector<float> b((size_t) N * K);
    for (auto & v : b) {
        v = dist(rng);
    }

    // CPU reference on 64 sampled output positions
    const struct ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    std::vector<sample> samples;
    {
        std::vector<float> row(K);
        for (int s = 0; s < 64; ++s) {
            const int m = (int) (rng() % M);
            const int n = (int) (rng() % N);
            traits->to_float(q.data() + (size_t) m * row_size, row.data(), K);
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double) row[k] * b[(size_t) n*K + k];
            }
            samples.push_back({ m, n, acc });
        }
    }

    cl_int err;
    cl_mem dQ = clCreateBuffer(context, CL_MEM_READ_ONLY,  q.size(),                     NULL, &err); CL_CHECK(err);
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) M * K * sizeof(float), NULL, &err); CL_CHECK(err);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY,  b.size() * sizeof(float),      NULL, &err); CL_CHECK(err);
    cl_mem dC = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) N * M * sizeof(float), NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, dQ, CL_TRUE, 0, q.size(), q.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, dB, CL_TRUE, 0, b.size() * sizeof(float), b.data(), 0, NULL, NULL));

    printf("K=%-5d M=%-5d N=%-4d %-22s\n", K, M, N, desc);
    const std::vector<float> zeros((size_t) N * M, 0.0f);

    // dequantization of the backend
    double dq_ms;
    {
        const cl_uint  nblocks = (cl_uint) ((size_t) M * nb);
        const cl_ulong zero    = 0;
        cl_kargs(be_k.dequant_q4_K).a(dQ).a(zero).a(dA).a(zero).a(nblocks);
        const size_t local  = 64;
        const size_t global = ((size_t) nblocks * 8 + local - 1) / local * local;
        dq_ms = time_kernel(pq, be_k.dequant_q4_K, 1, &global, &local, 3);
        printf("    dequant q4_K -> F32                %8.2f ms\n", dq_ms);
    }

    // matrix x vector kernel of the backend, once per column (the path of the short batches)
    {
        size_t l8_local   = 0;
        bool   l8_local_y = false;
        cl_kernel * kv = ggml_get_dmmv_l8_cl(GGML_TYPE_Q4_K, K, &l8_local, &l8_local_y);
        const size_t rows_per_wg = l8_local / 8;
        const size_t global      = ((size_t) M + rows_per_wg - 1) / rows_per_wg * l8_local;
        const cl_ulong zero = 0;
        cl_kargs args(*kv);
        args.a(dQ).a(zero).a(dB).a(zero).a(dC).a(zero).a((cl_int) K).a((cl_int) M).local(l8_local * sizeof(float));
        if (l8_local_y) {
            args.local((size_t) K * sizeof(float));
        }
        const double ms = time_kernel(pq, *kv, 1, &global, &l8_local, 5);
        printf("    dmmv q4_K, once per column         %8.2f ms (%.3f ms per column)\n", ms * N, ms);
    }

    // backend tiled kernel on the F32 weights
    {
        const cl_ulong zero = 0;
        cl_kargs(be_k.mul_mat_f_tiled).a(dA).a(zero).a(dB).a(zero).a(dC).a(zero)
                                      .a((cl_int) K).a((cl_int) M).a((cl_int) N).a((cl_int) 1)
                                      .a((cl_ulong) K*4).a((cl_ulong) M*K*4).a((cl_ulong) M*K*4)
                                      .a((cl_ulong) K*4).a((cl_ulong) N*K*4).a((cl_ulong) N*K*4)
                                      .a((cl_ulong) M*4).a((cl_ulong) N*M*4).a((cl_ulong) N*M*4)
                                      .a((cl_int) 1).a((cl_int) 1).a((cl_int) 0);
        const size_t local[3]  = { 16, 16, 1 };
        const size_t global[3] = { (size_t) (M + 63) / 64 * 16, (size_t) (N + 63) / 64 * 16, 1 };
        const double ms = time_kernel(pq, be_k.mul_mat_f_tiled, 3, global, local, 3);
        printf("    backend tiled 64x64 (F32)          %8.2f ms %7.1f GFLOPS | + dequant %8.2f ms | err %.1e\n",
               ms, flop / (ms * 1e-3) / 1e9, ms + dq_ms, max_rel_err(pq, dC, M, samples));
    }

    for (const gemm_cfg & c : cfgs) {
        cl_program p = build_with_defines(c);
        const size_t local[2]  = { (size_t) (c.BM / c.TM), (size_t) (c.BN / c.TN) };
        const size_t global[2] = { (size_t) (M + c.BM - 1) / c.BM * local[0], (size_t) (N + c.BN - 1) / c.BN * local[1] };

        size_t max_wg = 0;
        for (const char * kname : { "gemm_f32", "gemm_f32_u", "gemm_f32_u2", "gemm_f32_pf", "gemm_f32_v4m0", "gemm_f32_v4m1" }) {
            const bool pf = strcmp(kname, "gemm_f32_pf") == 0;
            const bool v4 = strncmp(kname, "gemm_f32_v4", 11) == 0;
            const bool u  = strncmp(kname, "gemm_f32_u", 10) == 0;
            const bool u2 = strcmp(kname, "gemm_f32_u2") == 0;
            const int  nthr = (c.BM / c.TM) * (c.BN / c.TN);
            if (pf && getenv("BENCH_GEMM_PF") == NULL) {
                continue;
            }
            if (strcmp(kname, "gemm_f32") == 0 && getenv("BENCH_GEMM_BASE") == NULL) {
                continue; // round 5: gemm_f32_u is 10-20% faster
            }
            if (u2 && M % c.BM != 0) {
                continue; // A is not padded here
            }
            if (u && (nthr % c.BK != 0 || (c.BM * c.BK) % nthr != 0 || (c.BN * c.BK) % nthr != 0 || K % c.BK != 0)) {
                continue;
            }
            if (!u && c.PAD != 2) {
                continue; // the other kernels do not use PAD: run them once per tile shape
            }
            if (v4 && (c.TM % 4 != 0 || c.TN % 4 != 0 || getenv("BENCH_GEMM_V4") == NULL)) {
                continue;
            }
            cl_kernel kf = clCreateKernel(p, kname, &err); CL_CHECK(err);
            cl_ulong priv = 0;
            cl_ulong lmem = 0;
            CL_CHECK(clGetKernelWorkGroupInfo(kf, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
            CL_CHECK(clGetKernelWorkGroupInfo(kf, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, NULL));
            CL_CHECK(clGetKernelWorkGroupInfo(kf, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL));
            const std::string label = u ? std::string(kname) + "/p" + std::to_string(c.PAD) : std::string(kname);
            if (local[0] * local[1] > max_wg) {
                printf("    %-13s BM%-3d BN%-3d BK%-2d TM%d TN%d: work-group %zu > max %zu\n", label.c_str(), c.BM, c.BN, c.BK, c.TM, c.TN, local[0]*local[1], max_wg);
            } else {
                // clear C, so that a kernel that does not write it cannot pass with the previous result
                CL_CHECK(clEnqueueWriteBuffer(pq, dC, CL_TRUE, 0, zeros.size() * sizeof(float), zeros.data(), 0, NULL, NULL));
                cl_kargs(kf).a(dA).a((cl_int) K).a(dB).a((cl_int) K).a(dC).a((cl_int) M).a((cl_int) M).a((cl_int) N).a((cl_int) K);
                const double ms = time_kernel(pq, kf, 2, global, local, 3);
                printf("    %-13s BM%-3d BN%-3d BK%-2d TM%d TN%d %8.2f ms %7.1f GFLOPS | + dequant %8.2f ms | private %3llu B, local %5llu B | err %.1e\n",
                       label.c_str(), c.BM, c.BN, c.BK, c.TM, c.TN, ms, flop / (ms * 1e-3) / 1e9, ms + dq_ms,
                       (unsigned long long) priv, (unsigned long long) lmem, max_rel_err(pq, dC, M, samples));
            }
            clReleaseKernel(kf);
        }

        if (c.BK == 16 && getenv("BENCH_GEMM_FUSED")) {
            cl_kernel kq = clCreateKernel(p, "gemm_q4_K", &err); CL_CHECK(err);
            CL_CHECK(clGetKernelWorkGroupInfo(kq, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
            if (local[0] * local[1] <= max_wg) {
                cl_kargs(kq).a(dQ).a((cl_int) nb).a(dB).a((cl_int) K).a(dC).a((cl_int) M).a((cl_int) M).a((cl_int) N).a((cl_int) K);
                const double ms = time_kernel(pq, kq, 2, global, local, 3);
                printf("    q4_K BM%-3d BN%-3d BK16 TM%d TN%d fused %8.2f ms %7.1f GFLOPS (no dequant pass)          | err %.1e\n",
                       c.BM, c.BN, c.TM, c.TN, ms, flop / (ms * 1e-3) / 1e9, max_rel_err(pq, dC, M, samples));
            }
            clReleaseKernel(kq);
        }
        clReleaseProgram(p);
    }

    clReleaseMemObject(dQ);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
}

// experimental Q4_K matrix x several vectors (round 8): like kernel_mmvn_q4_K, but every group of 8
// lanes handles R consecutive rows (every y value loaded feeds R FMAs, and y is re-read R times less),
// explicit fma chains instead of dot() + add, and the scales decoded per sub-block from the 3 packed
// words (fewer live registers than 16 floats per row). NC, R and LWG are defined before the source.
// Requires nrows % R == 0.
static const char * mmvx_source = R"CLC(
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void mmvx_q4_K(__global const uchar * xq, __global const float * yi, __global float * dst_base, const int doff,
               const int ldd, const int ncols, const int nrows, const int ny) {
    __local float tmp[NC*R*(LWG + 1)];
    __global float * dst = dst_base + doff;
    const int lid  = get_local_id(0);
    const int j    = lid & 7;
    const int row0 = (get_group_id(0) * (LWG/8) + (lid >> 3)) * R;
    const int nb   = ncols / 256;

    float acc[R][NC];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
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
                d[r]    = dm.x;
                dmin[r] = dm.y;
                __global const uint * sp = (__global const uint *) (blk + r*rs + 4);
                s0[r] = sp[0];
                s1[r] = sp[1];
                s2[r] = sp[2];
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
                        uint sc;
                        uint mn;
                        if (q < 4) {
                            sc = (s0[r] >> (8*q)) & 63;
                            mn = (s1[r] >> (8*q)) & 63;
                        } else {
                            sc = ((s2[r] >> (8*(q - 4)))     & 0xF) | (((s0[r] >> (8*(q - 4) + 6)) & 3) << 4);
                            mn = ((s2[r] >> (8*(q - 4) + 4)) & 0xF) | (((s1[r] >> (8*(q - 4) + 6)) & 3) << 4);
                        }
                        wf[r] = convert_float4(as_uchar4((w[r] >> (4*h)) & 0x0F0F0F0Fu)) * (d[r] * (float) sc) - dmin[r] * (float) mn;
                    }
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
            }
        }
    }
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
)CLC";

struct mmvx_cfg {
    int nc, r, lwg;
};

// short batches (round 7): kernel_mmvn_* of the backend (be_mmvn_source, NC columns per launch after
// kernel_mmvn_pack) against the matrix x vector kernel once per column and dequantization + GEMM
static void bench_mmvn(cl_command_queue pq, ggml_type type, int K, int M, const std::vector<int> & ns,
                       const std::vector<mmvx_cfg> & ncs, const std::vector<mmvx_cfg> & xcfgs, const char * desc) {
    const size_t row_size = ggml_row_size(type, K);
    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(row_size * M);
    {
        const int chunk_rows = 256;
        std::vector<float> chunk((size_t) chunk_rows * K);
        for (int r0 = 0; r0 < M; r0 += chunk_rows) {
            const int nr = std::min(chunk_rows, M - r0);
            for (size_t i = 0; i < (size_t) nr * K; ++i) {
                chunk[i] = dist(rng);
            }
            ggml_quantize_chunk(type, chunk.data(), q.data() + (size_t) r0 * row_size, 0, nr, K, nullptr);
        }
    }
    const int nmax = *std::max_element(ns.begin(), ns.end());
    std::vector<float> b((size_t) nmax * K);
    for (auto & v : b) {
        v = dist(rng);
    }
    // CPU reference: every column of 16 sampled rows
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<int> ref_rows;
    std::vector<double> ref; // ref[s*nmax + n]
    {
        std::vector<float> row(K);
        for (int s = 0; s < 16; ++s) {
            const int m = (int) (rng() % M);
            ref_rows.push_back(m);
            traits->to_float(q.data() + (size_t) m * row_size, row.data(), K);
            for (int n = 0; n < nmax; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (double) row[k] * b[(size_t) n*K + k];
                }
                ref.push_back(acc);
            }
        }
    }

    cl_int err;
    cl_mem dQ  = clCreateBuffer(context, CL_MEM_READ_ONLY,  q.size(),                        NULL, &err); CL_CHECK(err);
    cl_mem dB  = clCreateBuffer(context, CL_MEM_READ_ONLY,  b.size() * sizeof(float),         NULL, &err); CL_CHECK(err);
    cl_mem dC  = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) nmax * M * sizeof(float), NULL, &err); CL_CHECK(err);
    cl_mem dYi = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) 32 * K * sizeof(float),   NULL, &err); CL_CHECK(err);
    cl_mem dA  = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) M * K * sizeof(float),    NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, dQ, CL_TRUE, 0, q.size(), q.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, dB, CL_TRUE, 0, b.size() * sizeof(float), b.data(), 0, NULL, NULL));
    const std::vector<float> zeros((size_t) nmax * M, 0.0f);
    const cl_ulong zero = 0;

    // relative error of dC (N columns, column stride M) against the reference
    auto check = [&](int N) {
        double max_ref = 0.0;
        double max_err = 0.0;
        for (size_t s = 0; s < ref_rows.size(); ++s) {
            for (int n = 0; n < N; ++n) {
                float v;
                CL_CHECK(clEnqueueReadBuffer(pq, dC, CL_TRUE, ((size_t) n*M + ref_rows[s])*sizeof(float), sizeof(float), &v, 0, NULL, NULL));
                const double r = ref[s*nmax + n];
                max_ref = std::max(max_ref, std::fabs(r));
                max_err = std::max(max_err, std::fabs(v - r));
            }
        }
        return max_ref > 0.0 ? max_err / max_ref : max_err;
    };

    printf("%s K=%-5d M=%-5d %s\n", ggml_type_name(type), K, M, desc);

    // matrix x vector kernel once per column: time of one column
    double ms_col;
    {
        size_t l8_local   = 0;
        bool   l8_local_y = false;
        cl_kernel * kv = ggml_get_dmmv_l8_cl(type, K, &l8_local, &l8_local_y);
        const size_t rows_per_wg = l8_local / 8;
        const size_t global      = ((size_t) M + rows_per_wg - 1) / rows_per_wg * l8_local;
        cl_kargs args(*kv);
        args.a(dQ).a(zero).a(dB).a(zero).a(dC).a(zero).a((cl_int) K).a((cl_int) M).local(l8_local * sizeof(float));
        if (l8_local_y) {
            args.local((size_t) K * sizeof(float));
        }
        ms_col = time_kernel(pq, *kv, 1, &global, &l8_local, 5);
    }
    // dequantization (backend kernel) + kernel_gemm_f32 (256x32 tiles up to 32 columns, 128x64 above)
    cl_kernel kd = type == GGML_TYPE_Q4_K ? be_k.dequant_q4_K : type == GGML_TYPE_Q5_K ? be_k.dequant_q5_K : be_k.dequant_q6_K;
    const cl_uint nblocks = (cl_uint) ((size_t) M * K / ggml_blck_size(type));
    cl_kargs(kd).a(dQ).a(zero).a(dA).a(zero).a(nblocks);
    double ms_dq;
    {
        const size_t local  = 64;
        const size_t global = ((size_t) nblocks * 8 + local - 1) / local * local;
        ms_dq = time_kernel(pq, kd, 1, &global, &local, 3);
    }

    std::vector<cl_program> progs;
    std::vector<cl_kernel>  kpack, kmm;
    for (const mmvx_cfg & bc : ncs) {
        const std::string src = "#define NC " + std::to_string(bc.nc) + "\n#define R " + std::to_string(bc.r) +
                                "\n#define LWG " + std::to_string(bc.lwg) + "\n" + be_mmvn_source;
        cl_program p = build_program_from_source(context, device, src.c_str());
        progs.push_back(p);
        kpack.push_back(clCreateKernel(p, "kernel_mmvn_pack", &err)); CL_CHECK(err);
        const char * kname = type == GGML_TYPE_Q4_K ? "kernel_mmvn_q4_K" : type == GGML_TYPE_Q5_K ? "kernel_mmvn_q5_K" : "kernel_mmvn_q6_K";
        kmm.push_back(clCreateKernel(p, kname, &err)); CL_CHECK(err);
        cl_ulong priv = 0;
        CL_CHECK(clGetKernelWorkGroupInfo(kmm.back(), device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, NULL));
        printf("    NC=%-2d R=%d LWG=%d: private %llu B\n", bc.nc, bc.r, bc.lwg, (unsigned long long) priv);
    }
    // experimental variants (Q4_K only)
    std::vector<cl_program> xprogs;
    std::vector<cl_kernel>  xpack, xmm;
    if (type == GGML_TYPE_Q4_K) {
        for (const mmvx_cfg & x : xcfgs) {
            const std::string src = "#define NC " + std::to_string(x.nc) + "\n#define R " + std::to_string(x.r) +
                                    "\n#define LWG " + std::to_string(x.lwg) + "\n" + be_mmvn_source + mmvx_source;
            cl_program p = build_program_from_source(context, device, src.c_str());
            xprogs.push_back(p);
            xpack.push_back(clCreateKernel(p, "kernel_mmvn_pack", &err)); CL_CHECK(err);
            xmm.push_back(clCreateKernel(p, "mmvx_q4_K", &err)); CL_CHECK(err);
            cl_ulong priv = 0;
            CL_CHECK(clGetKernelWorkGroupInfo(xmm.back(), device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, NULL));
            printf("    x NC=%-2d R=%d LWG=%d: private %llu B\n", x.nc, x.r, x.lwg, (unsigned long long) priv);
        }
    }

    for (int N : ns) {
        printf("  N=%-3d dmmv x N %7.2f ms", N, ms_col * N);
        {
            const bool narrow = N <= be_gemm_n32.bn;
            const ggml_cl_be_gemm_shape & s = narrow ? be_gemm_n32 : be_gemm_n64;
            cl_kernel kg = narrow ? be_k.gemm_n32 : be_k.gemm_n64;
            cl_kargs(kg).a(dA).a((cl_int) K).a(dB).a(zero).a((cl_int) K).a(dC).a(zero).a((cl_int) M)
                        .a((cl_int) M).a((cl_int) N).a((cl_int) K);
            const size_t local[2]  = { (size_t) (s.bm / s.tm), (size_t) (s.bn / s.tn) };
            const size_t global[2] = { ((size_t) M + s.bm - 1) / s.bm * local[0], ((size_t) N + s.bn - 1) / s.bn * local[1] };
            const double ms = time_kernel(pq, kg, 2, global, local, 3);
            printf(" | dq+gemm %7.2f ms (err %.0e)", ms_dq + ms, check(N));
        }
        for (size_t v = 0; v < ncs.size(); ++v) {
            const int nc = ncs[v].nc;
            CL_CHECK(clEnqueueWriteBuffer(pq, dC, CL_TRUE, 0, zeros.size() * sizeof(float), zeros.data(), 0, NULL, NULL));
            double sum_ms = 0.0;
            const int iters = 3;
            for (int it = 0; it < iters + 1; ++it) {
                double ms = 0.0;
                for (int n0 = 0; n0 < N; n0 += nc) {
                    const int ny = std::min(nc, N - n0);
                    const cl_ulong yoff = (cl_ulong) n0 * K * sizeof(float);
                    const cl_ulong doff = (cl_ulong) n0 * M * sizeof(float);
                    cl_kargs(kpack[v]).a(dB).a(yoff).a((cl_int) K).a((cl_int) ny).a((cl_int) K).a(dYi);
                    const size_t lp = 256;
                    const size_t gp = ((size_t) K * nc + lp - 1) / lp * lp;
                    cl_event e1, e2;
                    CL_CHECK(clEnqueueNDRangeKernel(pq, kpack[v], 1, NULL, &gp, &lp, 0, NULL, &e1));
                    cl_kargs(kmm[v]).a(dQ).a(zero).a(dYi).a(dC).a(doff).a((cl_int) M).a((cl_int) K).a((cl_int) M).a((cl_int) ny);
                    const size_t lm = ncs[v].lwg;
                    const size_t rows_per_wg = (size_t) (ncs[v].lwg / 8) * ncs[v].r;
                    const size_t gm = ((size_t) M + rows_per_wg - 1) / rows_per_wg * lm;
                    CL_CHECK(clEnqueueNDRangeKernel(pq, kmm[v], 1, NULL, &gm, &lm, 0, NULL, &e2));
                    CL_CHECK(clWaitForEvents(1, &e2));
                    ms += event_ms(e1) + event_ms(e2);
                    CL_CHECK(clReleaseEvent(e1));
                    CL_CHECK(clReleaseEvent(e2));
                }
                if (it >= 1) {
                    sum_ms += ms;
                }
            }
            printf(" | %d/%d/%d %6.2f ms (err %.0e)", nc, ncs[v].r, ncs[v].lwg, sum_ms / iters, check(N));
        }
        for (size_t v = 0; v < xmm.size(); ++v) {
            const mmvx_cfg & x = xcfgs[v];
            CL_CHECK(clEnqueueWriteBuffer(pq, dC, CL_TRUE, 0, zeros.size() * sizeof(float), zeros.data(), 0, NULL, NULL));
            double sum_ms = 0.0;
            const int iters = 3;
            for (int it = 0; it < iters + 1; ++it) {
                double ms = 0.0;
                for (int n0 = 0; n0 < N; n0 += x.nc) {
                    const int ny = std::min(x.nc, N - n0);
                    const cl_ulong yoff = (cl_ulong) n0 * K * sizeof(float);
                    cl_kargs(xpack[v]).a(dB).a(yoff).a((cl_int) K).a((cl_int) ny).a((cl_int) K).a(dYi);
                    const size_t lp = 256;
                    const size_t gp = ((size_t) K * x.nc + lp - 1) / lp * lp;
                    cl_event e1, e2;
                    CL_CHECK(clEnqueueNDRangeKernel(pq, xpack[v], 1, NULL, &gp, &lp, 0, NULL, &e1));
                    cl_kargs(xmm[v]).a(dQ).a(dYi).a(dC).a((cl_int) (n0 * M)).a((cl_int) M).a((cl_int) K).a((cl_int) M).a((cl_int) ny);
                    const size_t lm = x.lwg;
                    const size_t rows_per_wg = (size_t) (x.lwg / 8) * x.r;
                    const size_t gm = ((size_t) M + rows_per_wg - 1) / rows_per_wg * lm;
                    CL_CHECK(clEnqueueNDRangeKernel(pq, xmm[v], 1, NULL, &gm, &lm, 0, NULL, &e2));
                    CL_CHECK(clWaitForEvents(1, &e2));
                    ms += event_ms(e1) + event_ms(e2);
                    CL_CHECK(clReleaseEvent(e1));
                    CL_CHECK(clReleaseEvent(e2));
                }
                if (it >= 1) {
                    sum_ms += ms;
                }
            }
            printf(" | x%d/%d/%d %6.2f ms (err %.0e)", x.nc, x.r, x.lwg, sum_ms / iters, check(N));
        }
        printf("\n");
    }
    for (size_t v = 0; v < xmm.size(); ++v) {
        clReleaseKernel(xpack[v]);
        clReleaseKernel(xmm[v]);
        clReleaseProgram(xprogs[v]);
    }
    for (size_t v = 0; v < ncs.size(); ++v) {
        clReleaseKernel(kpack[v]);
        clReleaseKernel(kmm[v]);
        clReleaseProgram(progs[v]);
    }
    clReleaseMemObject(dQ);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseMemObject(dYi);
    clReleaseMemObject(dA);
}

int main(void) {
    struct ggml_init_params params = { /* .mem_size = */ 1024*1024, /* .mem_buffer = */ NULL, /* .no_alloc = */ true };
    struct ggml_context * gctx = ggml_init(params);

    // the first OpenCL device of the registry (GGML_OPENCL_PLATFORM / GGML_OPENCL_DEVICE choose it)
    ggml_cl_probe_devices();
    if (g_cl_devices.empty() || !ggml_cl_activate(0)) {
        fprintf(stderr, "no OpenCL device\n");
        return 1;
    }

    cl_int err;
    cl_command_queue pq = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // FP32 peak of the device
    {
        cl_program p = build_program_from_source(context, device, peak_source);
        cl_kernel  k = clCreateKernel(p, "peak_fma", &err); CL_CHECK(err);
        const size_t local  = 256;
        const size_t global = 256 * 96;
        cl_mem dout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, global * sizeof(float), NULL, &err); CL_CHECK(err);
        cl_kargs(k).a(dout).a(0.999f);
        const double ms = time_kernel(pq, k, 1, &global, &local, 5);
        printf("peak fma: %.2f ms, %.1f GFLOPS\n", ms, (double) global * 1024 * 4 * 8 * 2 / (ms * 1e-3) / 1e9);
        clReleaseMemObject(dout);
        clReleaseKernel(k);
        clReleaseProgram(p);
    }

    // round 1 (2026-11-04): 64x64/16/4x4 83-97 GFLOPS, 128x64/16/8x4 104-110, 64x128/16/4x8 101-109,
    // 128x128/16/8x8 14 (register spills), 96x96/16/6x6 89-96, BK 8 worse; the fused Q4_K variant is
    // not better (the dequantization pass only takes 3-4% of the time).
    // round 2: reqd_work_group_size changes nothing; BK 32 is 2x slower (24 KB of local memory per
    // work-group); the register prefetch (gemm_f32_pf, BENCH_GEMM_PF=1) spills with 256-thread
    // work-groups and only matches the best one with 64x128/16/4x4 (512 threads): 103-110 GFLOPS
    // round 3: measured FP32 peak 234 GFLOPS (so 110 GFLOPS = 47%); the float4 variants
    // (gemm_f32_v4m0/m1, BENCH_GEMM_V4=1) are 10-30% slower; coalesced loads (loadmap 0) matter more
    // than the 2-way bank conflicts of the transposed stores; 128x128/16/8x4 = 128x64/16/8x4
    // round 5: unrolled gemm_f32_u (PAD 2) 122-131 GFLOPS with 128x64/16/8x4 (gemm_f32: 104-110),
    // 109-123 with 64x64/16/4x4; PAD 2 is 1-7% faster than PAD 1; TM6/TN6 tiles spill heavily
    const std::vector<gemm_cfg> cfgs_big = {
        { 128,  64, 16, 8, 4, 2 },
        {  64,  64, 16, 4, 4, 2 },
        {  64, 128, 16, 4, 8, 2 },
        { 128, 128, 16, 8, 4, 2 },
        { 128,  64, 16, 8, 4, 4 },
    };
    // round 4: narrow tiles for short batches; 256x32/16/8x4 is the best one with 32 columns
    const std::vector<gemm_cfg> cfgs_small = {
        { 256,  32, 16, 8, 4, 2 },
        { 128,  32, 16, 4, 4, 2 },
    };

    // BENCH_GEMM_DUMP_PTX=<prefix>: write the program binary (PTX on NVIDIA) of some configurations
    // to <prefix>-BMxBNxBK-TMxTN.ptx and exit
    if (const char * prefix = getenv("BENCH_GEMM_DUMP_PTX")) {
        for (const gemm_cfg & c : { gemm_cfg { 128, 64, 16, 8, 4 }, gemm_cfg { 64, 64, 16, 4, 4 } }) {
            cl_program p = build_with_defines(c);
            size_t bin_size = 0;
            CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL));
            std::vector<unsigned char> bin(bin_size);
            unsigned char * bin_ptr = bin.data();
            CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, NULL));
            const std::string path = std::string(prefix) + "-" + std::to_string(c.BM) + "x" + std::to_string(c.BN) + "x" +
                                     std::to_string(c.BK) + "-" + std::to_string(c.TM) + "x" + std::to_string(c.TN) + ".ptx";
            FILE * f = fopen(path.c_str(), "wb");
            if (f) {
                fwrite(bin.data(), 1, bin.size(), f);
                fclose(f);
                printf("program binary (%zu bytes) written to %s\n", bin.size(), path.c_str());
            }
            clReleaseProgram(p);
        }
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    // BENCH_GEMM_MMVN=1: only the short batches (matrix x several vectors)
    if (getenv("BENCH_GEMM_MMVN")) {
        // BENCH_GEMM_REGS=1: ptxas report (registers, spills) of kernel_mmvn_* for NC = 4, 8, 16; with
        // BENCH_GEMM_MMVN_PTX=<prefix> also the PTX of every NC to <prefix>-mmvn-NC.ptx
        if (getenv("BENCH_GEMM_REGS")) {
            const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-nv-verbose";
            for (int nc : { 4, 8, 12 }) {
                const std::string src = "#define NC " + std::to_string(nc) + "\n#define R 2\n#define LWG 128\n" + be_mmvn_source;
                const char * src_ptr = src.c_str();
                cl_program p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
                CL_CHECK(err);
                CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
                size_t log_size = 0;
                CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
                std::string log(log_size, '\0');
                CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL));
                printf("== NC=%d\n%s\n", nc, log.c_str());
                if (const char * prefix = getenv("BENCH_GEMM_MMVN_PTX")) {
                    size_t bin_size = 0;
                    CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL));
                    std::vector<unsigned char> bin(bin_size);
                    unsigned char * bin_ptr = bin.data();
                    CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, NULL));
                    const std::string path = std::string(prefix) + "-mmvn-" + std::to_string(nc) + ".ptx";
                    FILE * f = fopen(path.c_str(), "wb");
                    if (f) {
                        fwrite(bin.data(), 1, bin.size(), f);
                        fclose(f);
                    }
                }
                clReleaseProgram(p);
            }
            clReleaseCommandQueue(pq);
            ggml_free(gctx);
            return 0;
        }
        // round 7 (kernel_mmvn_*, 1 row per lane group): NC 8 is the best one; float4 pointers for y
        // (128-bit loads) are 1.6-2.5x slower than vload4 (split into scalar loads).
        // round 8 (mmvx_q4_K): fma chains + scales per sub-block +13-30%; 2 rows per lane group (R 2)
        // +40-70% (8 columns: 3.4-3.9 ms against 9.8-11.8 once per column); R 4 and NC 16 with R 2 spill
        const std::vector<int> ns = { 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 };
        const std::vector<mmvx_cfg> bcfgs = { { 2, 2, 128 }, { 4, 2, 128 }, { 8, 2, 128 }, { 12, 2, 128 } };
        const std::vector<mmvx_cfg> xcfgs = {};
        bench_mmvn(pq, GGML_TYPE_Q4_K, 1536, 8960, ns, bcfgs, xcfgs, "(Qwen2.5-1.5B ffn up)");
        bench_mmvn(pq, GGML_TYPE_Q4_K, 8960, 1536, ns, bcfgs, xcfgs, "(Qwen2.5-1.5B ffn down)");
        bench_mmvn(pq, GGML_TYPE_Q5_K, 1536, 8960, ns, bcfgs, xcfgs, "(ffn up shape)");
        bench_mmvn(pq, GGML_TYPE_Q6_K, 8960, 1536, ns, bcfgs, xcfgs, "(Qwen2.5-1.5B ffn down, Q6_K layers)");
        bench_mmvn(pq, GGML_TYPE_Q6_K, 1536, 8960, ns, bcfgs, xcfgs, "(ffn up shape)");
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    bench_shape(pq, 1536, 8960, 512, "(Qwen2.5-1.5B ffn up)",   cfgs_big);
    bench_shape(pq, 8960, 1536, 512, "(Qwen2.5-1.5B ffn down)", cfgs_big);
    bench_shape(pq, 2048, 6144, 512, "(Qwen3.5-2B ffn up)",     cfgs_big);
    bench_shape(pq, 1536, 8960,  32, "(Qwen2.5-1.5B ffn up)",   cfgs_small);

    clReleaseCommandQueue(pq);
    ggml_free(gctx);
    return 0;
}
