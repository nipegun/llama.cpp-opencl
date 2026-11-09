// bench-dmmv: micro-benchmark of the dequantize_mul_mat_vec (matrix x vector) kernels of the
// OpenCL backend, used for token generation.
//
// This file is built as a unity translation unit: it #includes ggml-opencl.cpp so that it uses
// exactly the same kernel sources, build options and kernel handles as the backend. It is linked
// against ggml-base only (target llama-bench-dmmv of OpenCL builds), so the OpenCL code only exists here.
//
// For every case it compares the original (legacy) kernel (still in the program source; the backend no
// longer creates it, see bench_old_dmmv_kernel) with the fast "l8" kernel selected by
// ggml_get_dmmv_l8_cl, reporting the kernel time
// measured with OpenCL profiling events as effective bandwidth (bytes of quantized weights read /
// kernel time) and the maximum relative error against a CPU reference (ggml to_float + dot
// product in double precision). It also measures the device read bandwidth as a ceiling.
// Set BENCH_DMMV_DUMP_PTX=<file> to write the program binary (PTX on NVIDIA) and exit.
//
// History of the kernel experiments on a GeForce GT 720M (Fermi GF117, OpenCL 1.1, 2026-11-04):
//   ceiling (streaming read kernel)                       10.26 GB/s
//   original (legacy) kernels                              1.4-2.6 GB/s
//   v2: one warp per super-block, 8 weights per lane       2.4-4.6 GB/s
//   v3: 8 lanes per super-block (32 weights per lane), vectorized scale decoding, 8 lanes per row
//       Q4_K 5.3-7.2, Q5_K 6.1-7.5, Q6_K (two aligned reads per unaligned word) 4.6-6.0 GB/s
//   v4l: y staged in local memory: no gain for Q4_K, +5-10% for Q6_K with ncols <= 2048
//   v4r2: 2 rows per lane group: slower (register pressure)
//   v5: y read through float4 pointers: slower than scalar loads
//   register caps: NOT applied. The OpenCL compiler of the 390.157 driver ignores
//       -cl-nv-maxrregcount=N silently (20, 24, 32, 40: the same registers) and rejects
//       "-cl-nv-maxrregcount N"; the conclusion "no gain" of the first round was wrong. Registers can
//       only be controlled through the code (see q4r QV below).
//   v6: "magic number" int->float conversion with pre-scaled y: no gain
// Round 2 (Qwen3.5-2B; numbers from the A/B modes, after gpu_warmup):
//   measurement: the clocks of the GT 720M rise only under sustained load, so the first measurements of
//       a run were up to 20-25% slower than the last ones; the A/B modes (BENCH_DMMV_Q4AB, _Q6AB) warm the
//       GPU up for 3 s and time every candidate in interleaved rounds (best time of each)
//   access pattern (BENCH_DMMV_PATTERN): reading the rows of a matrix like the kernels (8 lanes per row,
//       32 rows per work-group) reaches 9.8-10.1 GB/s, the same as tiles of 32 rows stored block-major:
//       the pattern is not the limit, the work of the kernels is
//   Q6_K q6x (BENCH_DMMV_Q6X): lane = 8 positions of one half (4 scales instead of 8), sums of 8 values of
//       y for the -32 offset, super-blocks in pairs (even ones 4-byte aligned): 6.3 -> 7.2 GB/s (output),
//       5.1 -> 6.6 (ffn_down, with 512 work-items: y of 6144 values takes 24 KB of local memory)
//   Q6_K q6r (BENCH_DMMV_Q6R): the same on rows repacked on the host (aligned 8-byte loads, 4 scales in one
//       32-bit load, 64 contiguous bytes per row and load) and y staged in chunks of 2048 values (local
//       memory independent of k): 8.5-8.7 (output), 8.2 (ffn_down), 7.6 (attn_v) GB/s with work-groups of
//       256 and 38 registers; unrolling 2 or 4 super-blocks (53-55 registers) and 2 rows per lane group
//       (59-63) are slower; several groups of rows per work-group (q6rt) change nothing
//       -> integrated: q6r layout (be_q6r_source, ggml_cl_q6r), llama-bench tg64 4.81 -> 5.52 t/s
//   Q4_K/Q5_K q45x (BENCH_DMMV_Q45X): y in local memory and the min term split among the lanes (lane j adds
//       the one of chunk j: 9 conversions instead of 16, no partial sums of y; 204 instructions per
//       super-block and lane instead of 239): +2-12%; with T groups of rows per work-group (q45xt, y in
//       one chunk) +5-6% more with 2048 columns and many rows
//   Q4_K q4r: rows repacked so that a lane owns a whole sub-block (one 128-bit load, 1 scale and 1 min,
//       162 instructions): 45 registers (the compiler hoists the 8 loads of y) -> 2 work-groups per
//       multiprocessor and 10% slower; with a loop of 2 words (QV 2) 40 registers and the same speed as
//       q45x, so no repacking for Q4_K
//       -> integrated: q45x/q45xt as kernel_dmmv_q4_K_ly1/_lyc, q5_K (be_q45_source), tg64 5.52 -> 5.68
// The integrated kernels are v3 (Q4_K, Q5_K, Q6_K with ncols > 2048) and v4l (Q6_K, ncols <= 2048) for
// the standard layout, and the q6r and q45 kernels of the backend (be_q6r_source, be_q45_source).
// Experiment modes: BENCH_DMMV_Q6X=1 (BENCH_DMMV_Q6X_REGS=1, BENCH_DMMV_Q6X_PTX=<prefix>), with
// BENCH_DMMV_Q6R=1 (BENCH_DMMV_Q6R_ONLY=1, BENCH_DMMV_Q6R_REGS=1); BENCH_DMMV_Q45X=1 (BENCH_DMMV_Q45X_REGS=1,
// BENCH_DMMV_Q45X_PTX=<prefix>, BENCH_DMMV_Q4R_OPTS="opts;opts"); BENCH_DMMV_Q4AB=1; BENCH_DMMV_Q6AB=1;
// BENCH_DMMV_PATTERN=1.

#include "ggml-opencl.cpp"

#include <cmath>
#include <random>
#include <string>

// A/B reference: the Q6_K v4l kernel exactly as it was written in the experiments, to compare it
// with the integrated dequantize_mul_mat_vec_q6_K_l8_ly on the same buffers
static const char * ref_v4l_source = MULTILINE_QUOTE(

inline float r_sum4(const float4 v) {
    return (v.x + v.y) + (v.z + v.w);
}

inline void r_row_reduce_store(__local float * tmp, const int lid, const int lrow, const float acc,
                               const int row, const int nrows, __global float * dst) {
    tmp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 8/2; s > 0; s >>= 1) {
        if (lrow < s) {
            tmp[lid] += tmp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lrow == 0 && row < nrows) {
        dst[row] = tmp[lid];
    }
}

inline uint r_ld_u32_a2(__global const uint * pa, const int idx, const uint sh, const uint hmask) {
    return (pa[idx] >> sh) | ((pa[idx + 1] << 16) & hmask);
}

inline void r_stage_y(__global const float * y, __local float * yl, const int ncols) {
    for (int c = get_local_id(0); c < ncols/4; c += get_local_size(0)) {
        vstore4(vload4(c, y), c, yl);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline float r_q6_group_l(const uint w, const uint hq, const int sel, __local const float * yp,
                          const float slo, const float shi) {
    const float4 ylo = vload4(0, yp);
    const float4 yhi = vload4(0, yp + 64);
    const float4 qlo = convert_float4(as_uchar4(( w       & 0x0F0F0F0Fu) | (((hq >> (2*sel))     & 0x03030303u) << 4)));
    const float4 qhi = convert_float4(as_uchar4(((w >> 4) & 0x0F0F0F0Fu) | (((hq >> (4 + 2*sel)) & 0x03030303u) << 4)));
    return slo * (dot(qlo, ylo) - 32.0f*r_sum4(ylo)) + shi * (dot(qhi, yhi) - 32.0f*r_sum4(yhi));
}

__kernel void ref_dmmv_q6_K_v4l(__global const uchar * xq, __global const float * y, __global float * dst,
                                const int ncols, const int nrows, __local float * tmp, __local float * yl) {
    r_stage_y(y, yl, ncols);
    const int lid  = get_local_id(0);
    const int lrow = lid % 8;
    const int row  = get_group_id(0) * (get_local_size(0) / 8) + lid / 8;
    const int j    = lrow & 7;
    const int nb   = ncols / QK_K;
    const int l    = 4*j;
    const int is   = j >> 2;

    float acc = 0.0f;
    if (row < nrows) {
        for (int i = 0; i < nb; i += 1) {
            const size_t boff  = ((size_t) row * nb + i) * 210;
            const uint   odd   = (uint) (boff & 2);
            const uint   sh    = odd * 8;
            const uint   hmask = 0u - (odd >> 1);
            __global const uint  * pa = (__global const uint *) (xq + boff - odd);
            __global const uchar * blk = xq + boff;
            __global const char  * sc  = (__global const char *) (blk + 192);
            __local  const float * yb  = yl + i*QK_K + l;
            const float d = vload_half(0, (__global const half *) (blk + 208));
            const uint h0 = r_ld_u32_a2(pa, (128 + l)/4, sh, hmask);
            const uint h1 = r_ld_u32_a2(pa, (160 + l)/4, sh, hmask);
            float sum = 0.0f;
            sum += r_q6_group_l(r_ld_u32_a2(pa, ( 0 + l)/4, sh, hmask), h0, 0, yb +   0, (float) sc[ 0 + is], (float) sc[ 4 + is]);
            sum += r_q6_group_l(r_ld_u32_a2(pa, (32 + l)/4, sh, hmask), h0, 1, yb +  32, (float) sc[ 2 + is], (float) sc[ 6 + is]);
            sum += r_q6_group_l(r_ld_u32_a2(pa, (64 + l)/4, sh, hmask), h1, 0, yb + 128, (float) sc[ 8 + is], (float) sc[12 + is]);
            sum += r_q6_group_l(r_ld_u32_a2(pa, (96 + l)/4, sh, hmask), h1, 1, yb + 160, (float) sc[10 + is], (float) sc[14 + is]);
            acc += d * sum;
        }
    }
    r_row_reduce_store(tmp, lid, lrow, acc, row, nrows, dst);
}

);

static cl_kernel ref_v4l_kernel = nullptr;

// ---------------------------------------------------------------------------------------------
// experimental Q6_K matrix x vector kernels (BENCH_DMMV_Q6X=1). Standard block_q6_K layout (210 bytes:
// ql[128], qh[64], scales[16], half d); requires an even number of super-blocks per row, so that
// every row starts 4-byte aligned, even super-blocks are 4-byte aligned and odd ones = 2 (mod 4).
// Lane j (8 lanes per row) handles half h = j >> 2 and the 8 positions l0 = 8*(j & 3) .. l0 + 7 of
// that half: ql[64h + l0 ..], ql[64h + 32 + l0 ..], qh[32h + l0 ..] -> 32 weights with only 4 scales.
// The -32 offset of the quants uses precomputed sums of every 8 values of y (ys, kernel q6x_ysum8).
// Compile-time parameters: LWG work-group size, R rows per lane group, YL y (and ys) staged in local
// memory (with 4 floats of padding every 128: no bank conflicts between the two halves), YV4 128-bit
// float4 loads of y, ODDLD 1 = 16-bit loads for the odd super-blocks (0 = three 32-bit loads).
// ---------------------------------------------------------------------------------------------
static const char * q6x_source = R"CLC(
#define M4 0x0F0F0F0Fu
#define M2 0x30303030u

#if YL
#define YSPACE __local
#define YOFF(c) ((c) + (((c) >> 7) << 2))
#else
#define YSPACE __global
#define YOFF(c) (c)
#endif

#if YV4
#define LDY(p, o) (((YSPACE const float4 *) ((p) + (o)))[0])
#else
#define LDY(p, o) vload4(0, (p) + (o))
#endif

__kernel void q6x_ysum8(__global const float * y, __global float * ys, const int n8) {
    const int m = get_global_id(0);
    if (m < n8) {
        const float4 a = vload4(0, y + 8*m);
        const float4 b = vload4(0, y + 8*m + 4);
        ys[m] = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
    }
}

inline float q6x_dot(const uint q, const float4 y) {
    const float4 f = convert_float4(as_uchar4(q));
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

// the 6-bit quants of the 4 groups of the lane (scales is, is + 2, is + 4, is + 6 of the half)
#define QG0(wl, wh, hq) (( (wl)       & M4) | (((hq) << 4) & M2))
#define QG1(wl, wh, hq) (( (wh)       & M4) | (((hq) << 2) & M2))
#define QG2(wl, wh, hq) ((((wl) >> 4) & M4) | ( (hq)       & M2))
#define QG3(wl, wh, hq) ((((wh) >> 4) & M4) | (((hq) >> 2) & M2))

// 6 words of the lane of a 4-byte aligned super-block
#define LD_A(b, wl0, wl1, wh0, wh1, hq0, hq1) { \
    __global const uint * pu_ = (__global const uint *) (b); \
    wl0 = pu_[oq4]; wl1 = pu_[oq4 + 1]; wh0 = pu_[oq4 + 8]; wh1 = pu_[oq4 + 9]; hq0 = pu_[oh4]; hq1 = pu_[oh4 + 1]; }

// 6 words of the lane of a super-block at 2 (mod 4)
#if ODDLD
#define U2(ps, i, v0, v1) { v0 = (uint) ps[i] | ((uint) ps[(i) + 1] << 16); v1 = (uint) ps[(i) + 2] | ((uint) ps[(i) + 3] << 16); }
#define LD_O(b, wl0, wl1, wh0, wh1, hq0, hq1) { \
    __global const ushort * ps_ = (__global const ushort *) (b); \
    U2(ps_, 2*oq4, wl0, wl1); U2(ps_, 2*oq4 + 16, wh0, wh1); U2(ps_, 2*oh4, hq0, hq1); }
#else
#define U2(pa, i, v0, v1) { const uint a0_ = pa[i]; const uint a1_ = pa[(i) + 1]; const uint a2_ = pa[(i) + 2]; \
    v0 = (a0_ >> 16) | (a1_ << 16); v1 = (a1_ >> 16) | (a2_ << 16); }
#define LD_O(b, wl0, wl1, wh0, wh1, hq0, hq1) { \
    __global const uint * pa_ = (__global const uint *) ((b) - 2); \
    U2(pa_, oq4, wl0, wl1); U2(pa_, oq4 + 8, wh0, wh1); U2(pa_, oh4, hq0, hq1); }
#endif

#define LD_SC(b, s0, s1, s2, s3) { \
    __global const char * sc_ = (__global const char *) (b) + 192 + osc; \
    s0 = (float) sc_[0]; s1 = (float) sc_[2]; s2 = (float) sc_[4]; s3 = (float) sc_[6]; }

// one group: 8 weights of one row, y at yb + o and yb + o + 4, ys value yg
#define GRP(QG, wl0, wl1, wh0, wh1, hq0, hq1, ya, yb_, yg, s, sum) \
    sum = fma(s, fma(-32.0f, yg, q6x_dot(QG(wl0, wh0, hq0), ya) + q6x_dot(QG(wl1, wh1, hq1), yb_)), sum)

#if R == 1
// one super-block of the row: b block pointer, LD load macro, yo element offset of the lane in y
#define BLOCK(b, LD, yo) { \
    uint wl0, wl1, wh0, wh1, hq0, hq1; float s0, s1, s2, s3; \
    LD(b, wl0, wl1, wh0, wh1, hq0, hq1); LD_SC(b, s0, s1, s2, s3); \
    const float d = vload_half(0, (__global const half *) ((b) + 208)); \
    YSPACE const float * yb = yv + YOFF(yo); \
    YSPACE const float * yq = ysv + ((yo) >> 3); \
    float sum = 0.0f; \
    GRP(QG0, wl0, wl1, wh0, wh1, hq0, hq1, LDY(yb,  0), LDY(yb,   4), yq[ 0], s0, sum); \
    GRP(QG1, wl0, wl1, wh0, wh1, hq0, hq1, LDY(yb, 32), LDY(yb,  36), yq[ 4], s1, sum); \
    GRP(QG2, wl0, wl1, wh0, wh1, hq0, hq1, LDY(yb, 64), LDY(yb,  68), yq[ 8], s2, sum); \
    GRP(QG3, wl0, wl1, wh0, wh1, hq0, hq1, LDY(yb, 96), LDY(yb, 100), yq[12], s3, sum); \
    acc0 = fma(d, sum, acc0); }
#else
// the same for two rows (blocks b and b + bstride) sharing the y loads
#define GRP2(QG, o, yg, sa, sb) { \
    const float4 ya = LDY(yb, o); const float4 yb4 = LDY(yb, (o) + 4); \
    GRP(QG, al0, al1, ah0, ah1, aq0, aq1, ya, yb4, yg, sa, suma); \
    GRP(QG, bl0, bl1, bh0, bh1, bq0, bq1, ya, yb4, yg, sb, sumb); }
#define BLOCK(b, LD, yo) { \
    uint al0, al1, ah0, ah1, aq0, aq1; float sa0, sa1, sa2, sa3; \
    uint bl0, bl1, bh0, bh1, bq0, bq1; float sb0, sb1, sb2, sb3; \
    LD(b, al0, al1, ah0, ah1, aq0, aq1); LD_SC(b, sa0, sa1, sa2, sa3); \
    LD((b) + bstride, bl0, bl1, bh0, bh1, bq0, bq1); LD_SC((b) + bstride, sb0, sb1, sb2, sb3); \
    const float da = vload_half(0, (__global const half *) ((b) + 208)); \
    const float db = vload_half(0, (__global const half *) ((b) + bstride + 208)); \
    YSPACE const float * yb = yv + YOFF(yo); \
    YSPACE const float * yq = ysv + ((yo) >> 3); \
    float suma = 0.0f; float sumb = 0.0f; \
    GRP2(QG0,  0, yq[ 0], sa0, sb0); \
    GRP2(QG1, 32, yq[ 4], sa1, sb1); \
    GRP2(QG2, 64, yq[ 8], sa2, sb2); \
    GRP2(QG3, 96, yq[12], sa3, sb3); \
    acc0 = fma(da, suma, acc0); acc1 = fma(db, sumb, acc1); }
#endif

__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void q6x(__global const uchar * xq, __global const float * y, __global const float * ys,
         __global float * dst, const int ncols, const int nrows, __local float * tmp
#if YL
         , __local float4 * yl4, __local float * ysl
#endif
         ) {
    const int lid = get_local_id(0);
    const int j   = lid & 7;
    const int h   = j >> 2;
    const int p   = j & 3;
    const int oq4 = 16*h + 2*p;
    const int oh4 = 32 + 8*h + 2*p;
    const int osc = 8*h + (p >> 1);
    const int nb  = ncols / QK_K;
    const int row0 = get_group_id(0) * (LWG/8 * R) + (lid >> 3);

#if YL
    __local float * yl = (__local float *) yl4;
    for (int c = lid; c < ncols/4; c += LWG) {
        vstore4(vload4(c, y), 0, yl + YOFF(4*c));
    }
    for (int c = lid; c < ncols/8; c += LWG) {
        ysl[c] = ys[c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local const float * yv  = yl;
    __local const float * ysv = ysl;
#else
    __global const float * yv  = y;
    __global const float * ysv = ys;
#endif

    const int ylane = 128*h + 8*p;
    float acc0 = 0.0f;
#if R == 1
    __global const uchar * b0 = xq + (size_t) min(row0, nrows - 1) * nb * 210;
#else
    float acc1 = 0.0f;
    const int row1 = row0 + LWG/8;
    __global const uchar * b0 = xq + (size_t) min(row0, nrows - 1) * nb * 210;
    const int bstride = (min(row1, nrows - 1) - min(row0, nrows - 1)) * nb * 210;
#endif
    for (int i = 0; i < nb; i += 2, b0 += 420) {
        BLOCK(b0,       LD_A, i*QK_K + ylane);
        BLOCK(b0 + 210, LD_O, i*QK_K + QK_K + ylane);
    }

    tmp[lid] = acc0;
#if R == 2
    tmp[LWG + lid] = acc1;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 4; s > 0; s >>= 1) {
        if (j < s) {
            tmp[lid] += tmp[lid + s];
#if R == 2
            tmp[LWG + lid] += tmp[LWG + lid + s];
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0) {
        if (row0 < nrows) {
            dst[row0] = tmp[lid];
        }
#if R == 2
        if (row1 < nrows) {
            dst[row1] = tmp[LWG + lid];
        }
#endif
    }
}
)CLC";

// q6r: the same kernel on Q6_K rows repacked on the host (q6r_repack_row), same row size. Every row of
// nb super-blocks becomes ql' [nb][2][8 lanes][8 bytes], qh' [nb][8 lanes][8 bytes], sc' [nb][4][4],
// d [nb]: every lane reads its quants with 8-byte loads, its 4 scales with one 32-bit load, and the
// 8 lanes of a row read contiguous 64-byte segments. The 4 ys values of a lane are reordered in local
// memory so that they are one float4. V2 1: uint2 loads (needs 8-byte aligned rows), 0: 2 x uint.
static const char * q6r_source = R"CLC(
#define M4 0x0F0F0F0Fu
#define M2 0x30303030u
#define YOFF(c) ((c) + (((c) >> 7) << 2))

inline float q6r_dot(const uint q, const float4 y) {
    const float4 f = convert_float4(as_uchar4(q));
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

#define QG0(wl, wh, hq) (( (wl)       & M4) | (((hq) << 4) & M2))
#define QG1(wl, wh, hq) (( (wh)       & M4) | (((hq) << 2) & M2))
#define QG2(wl, wh, hq) ((((wl) >> 4) & M4) | ( (hq)       & M2))
#define QG3(wl, wh, hq) ((((wh) >> 4) & M4) | (((hq) >> 2) & M2))

#if V2
#define LD2(p) (((__global const uint2 *) (p))[0])
#else
#define LD2(p) vload2(0, (__global const uint *) (p))
#endif
#define LDY(p, o) (((__local const float4 *) ((p) + (o)))[0])

// one group (8 weights) of one row: words a, b, c of the row, y at yi + o, ys value yg, scale s
#define GRP(a, b, c, QG, o, ya, yb4, yg, s, sum) \
    sum = fma(s, fma(-32.0f, yg, q6r_dot(QG(a.x, b.x, c.x), ya) + q6r_dot(QG(a.y, b.y, c.y), yb4)), sum)

// loads of one super-block of one row (pointers pq, ph, ps, pd of the row, block i of the chunk)
#define LDB(pq, ph, ps, pd, a, b, c, s, d) \
    const uint2 a = LD2(pq); const uint2 b = LD2((pq) + 64); const uint2 c = LD2(ph); \
    const char4 s = as_char4(*(ps)); const float d = vload_half(0, pd)

#if R == 1
#define BLOCK() { \
    LDB(pq0, ph0, ps0, pd0, a0, b0, c0, s0, d0); \
    const float4 yg = ysl4[8*i + 4*h + p]; \
    __local const float * yi = yl + YOFF(i*QK_K + 128*h + 8*p); \
    float sum0 = 0.0f; \
    { const float4 ya = LDY(yi,  0); const float4 yb4 = LDY(yi,   4); GRP(a0, b0, c0, QG0,  0, ya, yb4, yg.x, (float) s0.x, sum0); } \
    { const float4 ya = LDY(yi, 32); const float4 yb4 = LDY(yi,  36); GRP(a0, b0, c0, QG1, 32, ya, yb4, yg.y, (float) s0.y, sum0); } \
    { const float4 ya = LDY(yi, 64); const float4 yb4 = LDY(yi,  68); GRP(a0, b0, c0, QG2, 64, ya, yb4, yg.z, (float) s0.z, sum0); } \
    { const float4 ya = LDY(yi, 96); const float4 yb4 = LDY(yi, 100); GRP(a0, b0, c0, QG3, 96, ya, yb4, yg.w, (float) s0.w, sum0); } \
    acc0 = fma(d0, sum0, acc0); \
    pq0 += 128; ph0 += 64; ps0 += 4; pd0 += 1; }
#else
#define BLOCK() { \
    LDB(pq0, ph0, ps0, pd0, a0, b0, c0, s0, d0); \
    LDB(pq1, ph1, ps1, pd1, a1, b1, c1, s1, d1); \
    const float4 yg = ysl4[8*i + 4*h + p]; \
    __local const float * yi = yl + YOFF(i*QK_K + 128*h + 8*p); \
    float sum0 = 0.0f; float sum1 = 0.0f; \
    { const float4 ya = LDY(yi,  0); const float4 yb4 = LDY(yi,   4); \
      GRP(a0, b0, c0, QG0,  0, ya, yb4, yg.x, (float) s0.x, sum0); GRP(a1, b1, c1, QG0,  0, ya, yb4, yg.x, (float) s1.x, sum1); } \
    { const float4 ya = LDY(yi, 32); const float4 yb4 = LDY(yi,  36); \
      GRP(a0, b0, c0, QG1, 32, ya, yb4, yg.y, (float) s0.y, sum0); GRP(a1, b1, c1, QG1, 32, ya, yb4, yg.y, (float) s1.y, sum1); } \
    { const float4 ya = LDY(yi, 64); const float4 yb4 = LDY(yi,  68); \
      GRP(a0, b0, c0, QG2, 64, ya, yb4, yg.z, (float) s0.z, sum0); GRP(a1, b1, c1, QG2, 64, ya, yb4, yg.z, (float) s1.z, sum1); } \
    { const float4 ya = LDY(yi, 96); const float4 yb4 = LDY(yi, 100); \
      GRP(a0, b0, c0, QG3, 96, ya, yb4, yg.w, (float) s0.w, sum0); GRP(a1, b1, c1, QG3, 96, ya, yb4, yg.w, (float) s1.w, sum1); } \
    acc0 = fma(d0, sum0, acc0); acc1 = fma(d1, sum1, acc1); \
    pq0 += 128; ph0 += 64; ps0 += 4; pd0 += 1; \
    pq1 += 128; ph1 += 64; ps1 += 4; pd1 += 1; }
#endif

// row pointers of the q6r layout
#define ROWPTRS(row, pq, ph, ps, pd) \
    __global const uchar * rb_##pq = xq + (size_t) min(row, nrows - 1) * nb * 210; \
    __global const uchar * pq = rb_##pq + 8*j; \
    __global const uchar * ph = rb_##pq + nb*128 + 8*j; \
    __global const uint  * ps = (__global const uint *) (rb_##pq + nb*192) + 2*h + (p >> 1); \
    __global const half  * pd = (__global const half *) (rb_##pq + nb*208)

__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void q6r(__global const uchar * xq, __global const float * y, __global const float * ys,
         __global float * dst, const int ncols, const int nrows, __local float * tmp,
         __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int j   = lid & 7;
    const int h   = j >> 2;
    const int p   = j & 3;
    const int nb  = ncols / QK_K;
    const int row0 = get_group_id(0) * (LWG/8*R) + (lid >> 3);

    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;

    ROWPTRS(row0, pq0, ph0, ps0, pd0);
    float acc0 = 0.0f;
#if R == 2
    const int row1 = row0 + LWG/8;
    ROWPTRS(row1, pq1, ph1, ps1, pd1);
    float acc1 = 0.0f;
#endif

    // y (and ys) are staged in local memory in chunks of YCH values (YCH 0: the whole row)
    const int ych = YCH > 0 ? YCH : ncols;
    for (int k0 = 0; k0 < ncols; k0 += ych) {
        const int len = min(ych, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int c = lid; c < len/4; c += LWG) {
            vstore4(vload4(0, y + k0 + 4*c), 0, yl + YOFF(4*c));
        }
        for (int c = lid; c < len/8; c += LWG) {
            ysl[(c & ~15) | ((c & 3) << 2) | ((c >> 2) & 3)] = ys[k0/8 + c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nbc = len / QK_K;
#if UNR == 1
        for (int i = 0; i < nbc; ++i) {
            BLOCK();
        }
#elif UNR == 2
#pragma unroll 2
        for (int i = 0; i < nbc; ++i) {
            BLOCK();
        }
#else
#pragma unroll 4
        for (int i = 0; i < nbc; ++i) {
            BLOCK();
        }
#endif
    }

    tmp[lid] = acc0;
#if R == 2
    tmp[LWG + lid] = acc1;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 4; s > 0; s >>= 1) {
        if (j < s) {
            tmp[lid] += tmp[lid + s];
#if R == 2
            tmp[LWG + lid] += tmp[LWG + lid + s];
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0) {
        if (row0 < nrows) {
            dst[row0] = tmp[lid];
        }
#if R == 2
        if (row1 < nrows) {
            dst[row1] = tmp[LWG + lid];
        }
#endif
    }
}
)CLC";

// ---------------------------------------------------------------------------------------------
// experimental Q4_K / Q5_K matrix x vector kernels (BENCH_DMMV_Q45X=1), standard block layout, same lane
// layout as the integrated l8 kernels (lane j: 4 bytes of every quant word, weights 4j..4j+3 of every
// chunk of 32). y is staged in local memory in chunks of YCH values together with the sums of every 8
// values; the min term of the super-block, dmin * sum_q m_q * (sum of y over chunk q), is split among the
// lanes: lane j adds the one of chunk j, so it decodes only its own min. YV4: float4 loads of y.
// ---------------------------------------------------------------------------------------------
static const char * q45x_source = R"CLC(
#define M4 0x0F0F0F0Fu
#if YV4
#define LDY(p, o) (((__local const float4 *) ((p) + (o)))[0])
#else
#define LDY(p, o) vload4(0, (p) + (o))
#endif

inline float q45x_dot(const uint q, const float4 y) {
    const float4 f = convert_float4(as_uchar4(q));
    return fma(f.w, y.w, fma(f.z, y.z, fma(f.y, y.y, f.x * y.x)));
}

__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void q45x(__global const uchar * xq, __global const float * y, __global float * dst,
          const int ncols, const int nrows, __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int j   = lid & 7;
    const int nb  = ncols / QK_K;
    const int row = get_group_id(0) * (LWG/8) + (lid >> 3);
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    __global const uchar * blk = xq + (size_t) min(row, nrows - 1) * nb * BS;
    const int jj = j & 3;

    float acc = 0.0f;
    for (int k0 = 0; k0 < ncols; k0 += YCH) {
        const int len = min(YCH, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int e = lid; e < len/8; e += LWG) {
            const float4 u = vload4(0, y + k0 + 8*e);
            const float4 w = vload4(0, y + k0 + 8*e + 4);
            vstore4(u, 0, yl + 8*e);
            vstore4(w, 0, yl + 8*e + 4);
            ysl[e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nbc = len / QK_K;
        for (int i = 0; i < nbc; ++i, blk += BS) {
            const float2 dm = vload_half2(0, (__global const half *) blk);
            __global const uint * sp = (__global const uint *) (blk + 4);
            const uint s0 = sp[0];
            const uint s1 = sp[1];
            const uint s2 = sp[2];
            const float4 sc_a = convert_float4(as_uchar4(s0 & 0x3F3F3F3Fu));
            const float4 sc_b = convert_float4(as_uchar4((s2 & M4) | ((s0 >> 2) & 0x30303030u)));
            const uint   mlo  = (s1 >> (8*jj)) & 63;
            const uint   mhi  = ((s2 >> (8*jj + 4)) & 0xF) | (((s1 >> (8*jj + 6)) & 3) << 4);
            const float  mj   = (float) (j < 4 ? mlo : mhi);
            const float4 ys4  = ysl4[8*i + j];
            const float  ysj  = (ys4.x + ys4.y) + (ys4.z + ys4.w);
#if BS == 144
            __global const uint * qs = (__global const uint *) (blk + 16);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
#define QLO(w, q) ((w) & M4)
#define QHI(w, q) (((w) >> 4) & M4)
#else
            const uint hq = ((__global const uint *) (blk + 16))[j];
            __global const uint * qs = (__global const uint *) (blk + 48);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
#define QLO(w, q) (( (w)       & M4) | (((hq >> (q)) & 0x01010101u) << 4))
#define QHI(w, q) ((((w) >> 4) & M4) | (((hq >> (q)) & 0x01010101u) << 4))
#endif
            __local const float * yb = yl + i*QK_K + 4*j;
            float sd = 0.0f;
            sd = fma(sc_a.x, q45x_dot(QLO(w0, 0), LDY(yb,   0)), sd);
            sd = fma(sc_a.y, q45x_dot(QHI(w0, 1), LDY(yb,  32)), sd);
            sd = fma(sc_a.z, q45x_dot(QLO(w1, 2), LDY(yb,  64)), sd);
            sd = fma(sc_a.w, q45x_dot(QHI(w1, 3), LDY(yb,  96)), sd);
            sd = fma(sc_b.x, q45x_dot(QLO(w2, 4), LDY(yb, 128)), sd);
            sd = fma(sc_b.y, q45x_dot(QHI(w2, 5), LDY(yb, 160)), sd);
            sd = fma(sc_b.z, q45x_dot(QLO(w3, 6), LDY(yb, 192)), sd);
            sd = fma(sc_b.w, q45x_dot(QHI(w3, 7), LDY(yb, 224)), sd);
            acc = fma(dm.x, sd, fma(-dm.y * mj, ysj, acc));
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
)CLC";

// q4r: Q4_K rows repacked on the host (q4r_repack_row), same row size 144*nb: dm [nb][2 halves],
// sc' [nb][12], qs' [nb][8 sub-blocks][16 bytes]. Lane q (0..7) handles sub-block q (32 consecutive weights):
// its 16 quant bytes hold weights 2b (low nibble) and 2b + 1 (high nibble) of the sub-block in byte b, so one
// 128-bit load brings all its quants and the 8 lanes of a row read 128 contiguous bytes. sc' byte q = scale q
// (6 bits) | (min q & 3) << 6, nibble q of bytes 8..11 = min q >> 2. y in local memory with 4 floats of
// padding every 32 (the 8 lanes read 8 different sub-blocks: no bank conflicts).
static const char * q4r_source = R"CLC(
#define M4 0x0F0F0F0Fu
#define YOFF(c) ((c) + (((c) >> 5) << 2))
#define LDY(p, o) (((__local const float4 *) ((p) + (o)))[0])

// 8 weights of word w: low nibbles = even positions (ya.x, ya.z, yb.x, yb.z), high = odd ones
inline float q4r_dot8(const uint w, const float4 ya, const float4 yb) {
    const float4 lo = convert_float4(as_uchar4(w & M4));
    const float4 hi = convert_float4(as_uchar4((w >> 4) & M4));
    float s = lo.x * ya.x;
    s = fma(lo.y, ya.z, s);
    s = fma(lo.z, yb.x, s);
    s = fma(lo.w, yb.z, s);
    s = fma(hi.x, ya.y, s);
    s = fma(hi.y, ya.w, s);
    s = fma(hi.z, yb.y, s);
    s = fma(hi.w, yb.w, s);
    return s;
}

__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void q4r(__global const uchar * xq, __global const float * y, __global float * dst,
         const int ncols, const int nrows, __local float * tmp, __local float4 * yl4, __local float4 * ysl4) {
    const int lid = get_local_id(0);
    const int q   = lid & 7;
    const int nb  = ncols / QK_K;
    const int row = get_group_id(0) * (LWG/8) + (lid >> 3);
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 144;
    __global const half  * pdm = (__global const half *) rb;
    __global const uchar * psc = rb + 4*nb;
    __global const uint4 * pqs = (__global const uint4 *) (rb + 16*nb) + q;
    const int shb = 8*(q & 3);
    const int shn = 4*q;

    float acc = 0.0f;
    for (int k0 = 0; k0 < ncols; k0 += YCH) {
        const int len = min(YCH, ncols - k0);
        if (k0 > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int e = lid; e < len/8; e += LWG) {
            const float4 u = vload4(0, y + k0 + 8*e);
            const float4 w = vload4(0, y + k0 + 8*e + 4);
            vstore4(u, 0, yl + YOFF(8*e));
            vstore4(w, 0, yl + YOFF(8*e) + 4);
            ysl[e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nbc = len / QK_K;
        const int ib0 = k0 / QK_K;
        for (int i = 0; i < nbc; ++i) {
            const int    ib = ib0 + i;
            const float2 dm = vload_half2(ib, pdm);
            const uint   u0 = ((__global const uint *) (psc + 12*ib))[q >> 2];
            const uint   u2 = ((__global const uint *) (psc + 12*ib))[2];
            const uint   bq = (u0 >> shb) & 0xFF;
            const float  sc = (float) (bq & 63);
            const float  mn = (float) ((bq >> 6) | (((u2 >> shn) & 15) << 2));
            const float4 ys4 = ysl4[8*i + q];
            const float  ysq = (ys4.x + ys4.y) + (ys4.z + ys4.w);
            __local const float * yb = yl + YOFF(i*QK_K + 32*q);
#if QV == 4
            const uint4  w  = pqs[8*ib];
            float s = q4r_dot8(w.x, LDY(yb,  0), LDY(yb,  4));
            s += q4r_dot8(w.y, LDY(yb,  8), LDY(yb, 12));
            s += q4r_dot8(w.z, LDY(yb, 16), LDY(yb, 20));
            s += q4r_dot8(w.w, LDY(yb, 24), LDY(yb, 28));
#elif QV == 2
            __global const uint2 * pw = (__global const uint2 *) (pqs + 8*ib);
            float s = 0.0f;
#pragma unroll 1
            for (int t = 0; t < 2; ++t) {
                const uint2 w = pw[t];
                s += q4r_dot8(w.x, LDY(yb, 16*t + 0), LDY(yb, 16*t +  4));
                s += q4r_dot8(w.y, LDY(yb, 16*t + 8), LDY(yb, 16*t + 12));
            }
#else
            __global const uint * pw = (__global const uint *) (pqs + 8*ib);
            float s = 0.0f;
#pragma unroll 1
            for (int t = 0; t < 4; ++t) {
                s += q4r_dot8(pw[t], LDY(yb, 8*t), LDY(yb, 8*t + 4));
            }
#endif
            acc = fma(dm.x * sc, s, fma(-dm.y * mn, ysq, acc));
        }
    }

    tmp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = 4; st > 0; st >>= 1) {
        if (q < st) {
            tmp[lid] += tmp[lid + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (q == 0 && row < nrows) {
        dst[row] = tmp[lid];
    }
}
)CLC";

// q4rt: q4r (QV 2) where every work-group handles T groups of LWG/8 rows one after another with y staged
// once (ncols <= YCH): the staging of y and its barrier are paid once per T groups of rows
static const char * q4rt_source = R"CLC(
__kernel __attribute__((reqd_work_group_size(LWG, 1, 1)))
void q4rt(__global const uchar * xq, __global const float * y, __global float * dst,
          const int ncols, const int nrows, __local float * tmp, __local float4 * yl4, __local float4 * ysl4, const int T) {
    const int lid = get_local_id(0);
    const int q   = lid & 7;
    const int nb  = ncols / QK_K;
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    const int shb = 8*(q & 3);
    const int shn = 4*q;

    for (int e = lid; e < ncols/8; e += LWG) {
        const float4 u = vload4(0, y + 8*e);
        const float4 w = vload4(0, y + 8*e + 4);
        vstore4(u, 0, yl + YOFF(8*e));
        vstore4(w, 0, yl + YOFF(8*e) + 4);
        ysl[e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (LWG/8) + (lid >> 3);
        __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 144;
        __global const half  * pdm = (__global const half *) rb;
        __global const uchar * psc = rb + 4*nb;
        __global const uint4 * pqs = (__global const uint4 *) (rb + 16*nb) + q;
        float acc = 0.0f;
        for (int i = 0; i < nb; ++i) {
            const float2 dm = vload_half2(i, pdm);
            const uint   u0 = ((__global const uint *) (psc + 12*i))[q >> 2];
            const uint   u2 = ((__global const uint *) (psc + 12*i))[2];
            const uint   bq = (u0 >> shb) & 0xFF;
            const float  sc = (float) (bq & 63);
            const float  mn = (float) ((bq >> 6) | (((u2 >> shn) & 15) << 2));
            const float4 ys4 = ysl4[8*i + q];
            const float  ysq = (ys4.x + ys4.y) + (ys4.z + ys4.w);
            __local const float * yb = yl + YOFF(i*QK_K + 32*q);
            __global const uint2 * pw = (__global const uint2 *) (pqs + 8*i);
            float s = 0.0f;
#pragma unroll 1
            for (int t = 0; t < 2; ++t) {
                const uint2 w = pw[t];
                s += q4r_dot8(w.x, LDY(yb, 16*t + 0), LDY(yb, 16*t +  4));
                s += q4r_dot8(w.y, LDY(yb, 16*t + 8), LDY(yb, 16*t + 12));
            }
            acc = fma(dm.x * sc, s, fma(-dm.y * mn, ysq, acc));
        }
        tmp[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int st = 4; st > 0; st >>= 1) {
            if (q < st) {
                tmp[lid] += tmp[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (q == 0 && row < nrows) {
            dst[row] = tmp[lid];
        }
    }
}
)CLC";

// standard block_q4_K row (nb blocks of half d, half dmin, scales[12], qs[128]) -> q4r layout
static void q4r_repack_row(const uint8_t * src, uint8_t * dst, int nb) {
    uint8_t * dmp = dst;
    uint8_t * scp = dst + (size_t) nb * 4;
    uint8_t * qsp = dst + (size_t) nb * 16;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * b  = src + (size_t) i * 144;
        const uint8_t * s  = b + 4;
        const uint8_t * qs = b + 16;
        memcpy(dmp + 4*i, b, 4);
        uint8_t sc[8];
        uint8_t mn[8];
        for (int k = 0; k < 8; ++k) {
            if (k < 4) {
                sc[k] = s[k] & 63;
                mn[k] = s[k + 4] & 63;
            } else {
                sc[k] = (s[k + 4] & 0xF) | ((s[k - 4] >> 6) << 4);
                mn[k] = (s[k + 4] >>  4) | ((s[k    ] >> 6) << 4);
            }
        }
        uint8_t * o = scp + (size_t) i * 12;
        for (int k = 0; k < 8; ++k) {
            o[k] = (uint8_t) (sc[k] | ((mn[k] & 3) << 6));
        }
        for (int k = 0; k < 4; ++k) {
            o[8 + k] = (uint8_t) ((mn[2*k] >> 2) | ((mn[2*k + 1] >> 2) << 4));
        }
        // weight l (0..31) of sub-block k: nibble of qs[32*(k/2) + l], low for even k, high for odd k
        for (int k = 0; k < 8; ++k) {
            for (int bb = 0; bb < 16; ++bb) {
                const int l0 = 2*bb;
                const int l1 = 2*bb + 1;
                const uint8_t v0 = (k & 1) ? (qs[32*(k/2) + l0] >> 4) : (qs[32*(k/2) + l0] & 0xF);
                const uint8_t v1 = (k & 1) ? (qs[32*(k/2) + l1] >> 4) : (qs[32*(k/2) + l1] & 0xF);
                qsp[(size_t) i*128 + 16*k + bb] = (uint8_t) (v0 | (v1 << 4));
            }
        }
    }
}

// q45xt: q45x (BS 144 or 176, YV4 0) where every work-group handles T groups of rows with y staged once
// (ncols <= 2048, one chunk)
static const char * q45xt_source = R"CLC(
__kernel void q45xt(__global const uchar * xq, __global const float * y, __global float * dst,
                    const int ncols, const int nrows, __local float * tmp, __local float4 * yl4, __local float4 * ysl4, const int T) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int nb  = ncols / QK_K;
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    const int jj = j & 3;
    for (int e = lid; e < ncols/8; e += lsz) {
        const float4 u = vload4(0, y + 8*e);
        const float4 w = vload4(0, y + 8*e + 4);
        vstore4(u, 0, yl + 8*e);
        vstore4(w, 0, yl + 8*e + 4);
        ysl[e] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (lsz >> 3) + (lid >> 3);
        __global const uchar * blk = xq + (size_t) min(row, nrows - 1) * nb * BS;
        float acc = 0.0f;
        for (int i = 0; i < nb; ++i, blk += BS) {
            const float2 dm = vload_half2(0, (__global const half *) blk);
            __global const uint * sp = (__global const uint *) (blk + 4);
            const uint s0 = sp[0];
            const uint s1 = sp[1];
            const uint s2 = sp[2];
            const float4 sc_a = convert_float4(as_uchar4(s0 & 0x3F3F3F3Fu));
            const float4 sc_b = convert_float4(as_uchar4((s2 & M4) | ((s0 >> 2) & 0x30303030u)));
            const uint   mlo  = (s1 >> (8*jj)) & 63;
            const uint   mhi  = ((s2 >> (8*jj + 4)) & 0xF) | (((s1 >> (8*jj + 6)) & 3) << 4);
            const float  mj   = (float) (j < 4 ? mlo : mhi);
            const float4 ys4  = ysl4[8*i + j];
            const float  ysj  = (ys4.x + ys4.y) + (ys4.z + ys4.w);
#if BS == 144
            __global const uint * qs = (__global const uint *) (blk + 16);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
#else
            const uint hq = ((__global const uint *) (blk + 16))[j];
            __global const uint * qs = (__global const uint *) (blk + 48);
            const uint w0 = qs[j];
            const uint w1 = qs[j +  8];
            const uint w2 = qs[j + 16];
            const uint w3 = qs[j + 24];
#endif
            __local const float * yb = yl + i*QK_K + 4*j;
            float sd = 0.0f;
            sd = fma(sc_a.x, q45x_dot(QLO(w0, 0), LDY(yb,   0)), sd);
            sd = fma(sc_a.y, q45x_dot(QHI(w0, 1), LDY(yb,  32)), sd);
            sd = fma(sc_a.z, q45x_dot(QLO(w1, 2), LDY(yb,  64)), sd);
            sd = fma(sc_a.w, q45x_dot(QHI(w1, 3), LDY(yb,  96)), sd);
            sd = fma(sc_b.x, q45x_dot(QLO(w2, 4), LDY(yb, 128)), sd);
            sd = fma(sc_b.y, q45x_dot(QHI(w2, 5), LDY(yb, 160)), sd);
            sd = fma(sc_b.z, q45x_dot(QLO(w3, 6), LDY(yb, 192)), sd);
            sd = fma(sc_b.w, q45x_dot(QHI(w3, 7), LDY(yb, 224)), sd);
            acc = fma(dm.x, sd, fma(-dm.y * mj, ysj, acc));
        }
        tmp[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int st = 4; st > 0; st >>= 1) {
            if (j < st) {
                tmp[lid] += tmp[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (j == 0 && row < nrows) {
            dst[row] = tmp[lid];
        }
    }
}
)CLC";

struct q45x_variant {
    int lwg;
    int yv4;
    int ych;
};

static const q45x_variant q45x_variants[] = {
    { 128, 1, 2048 },
    { 256, 1, 2048 },
    { 256, 0, 2048 },
    { 256, 1, 1024 },
    { 512, 1, 2048 },
};

// standard block_q6_K row (nb blocks of ql[128], qh[64], scales[16], half d) -> q6r layout
static void q6r_repack_row(const uint8_t * src, uint8_t * dst, int nb) {
    uint8_t * qlp = dst;
    uint8_t * qhp = dst + (size_t) nb * 128;
    uint8_t * scp = dst + (size_t) nb * 192;
    uint8_t * dp  = dst + (size_t) nb * 208;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * b  = src + (size_t) i * 210;
        const uint8_t * ql = b;
        const uint8_t * qh = b + 128;
        const uint8_t * sc = b + 192;
        for (int j = 0; j < 8; ++j) {
            const int h = j >> 2;
            const int p = j & 3;
            memcpy(qlp + i*128 +      8*j, ql + 64*h +      8*p, 8);
            memcpy(qlp + i*128 + 64 + 8*j, ql + 64*h + 32 + 8*p, 8);
            memcpy(qhp + i*64  +      8*j, qh + 32*h +      8*p, 8);
        }
        for (int hh = 0; hh < 2; ++hh) {
            for (int is = 0; is < 2; ++is) {
                for (int g = 0; g < 4; ++g) {
                    scp[i*16 + 4*(2*hh + is) + g] = sc[8*hh + is + 2*g];
                }
            }
        }
        memcpy(dp + 2*i, b + 208, 2);
    }
}

struct q6r_variant {
    int lwg;
    int v2;
    int ych; // 0: the whole row
    int r;
    int unr;
};

static const q6r_variant q6r_variants[] = {
    { 256, 1,    0, 1, 2 },
    { 512, 1,    0, 1, 2 },
    { 256, 1, 2048, 1, 2 },
    { 128, 1, 2048, 1, 2 },
    { 256, 1, 1024, 1, 2 },
    { 256, 1, 2048, 1, 1 },
    { 256, 1, 2048, 1, 4 },
    { 256, 1, 2048, 2, 1 },
    { 128, 1, 2048, 2, 1 },
    { 256, 1, 2048, 2, 2 },
};

struct q6x_variant {
    int lwg;
    int r;
    int yl;
    int yv4;
    int oddld;
};

static std::string q6x_defines(const q6x_variant & v) {
    return "#define LWG " + std::to_string(v.lwg) + "\n#define R " + std::to_string(v.r) +
           "\n#define YL " + std::to_string(v.yl) + "\n#define YV4 " + std::to_string(v.yv4) +
           "\n#define ODDLD " + std::to_string(v.oddld) + "\n";
}

static const q6x_variant q6x_variants[] = {
    { 256, 1, 1, 0, 0 },
    { 256, 1, 1, 1, 0 },
    { 256, 1, 1, 1, 1 },
    { 256, 2, 1, 1, 0 },
    { 128, 1, 1, 1, 0 },
    { 512, 1, 1, 1, 0 },
    { 128, 1, 0, 0, 0 },
    { 128, 1, 0, 1, 0 },
    { 256, 1, 0, 0, 0 },
    { 128, 2, 0, 0, 0 },
};

struct bench_case {
    ggml_type    type;
    int64_t      ncols;
    int64_t      nrows;
    int          iters;
    const char * desc;
};

static double event_ms(cl_event ev) {
    cl_ulong t0 = 0;
    cl_ulong t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL));
    return (double) (t1 - t0) * 1e-6;
}

// runs a kernel whose arguments are already set; returns the average kernel time in ms
static double time_kernel(cl_command_queue pq, cl_kernel k, size_t global, size_t local, int iters) {
    double sum_ms = 0.0;
    for (int it = 0; it < iters + 2; ++it) {
        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(pq, k, 1, NULL, &global, &local, 0, NULL, &ev));
        CL_CHECK(clWaitForEvents(1, &ev));
        if (it >= 2) { // first two runs are warm-up
            sum_ms += event_ms(ev);
        }
        CL_CHECK(clReleaseEvent(ev));
    }
    return sum_ms / iters;
}

static double max_rel_err(const std::vector<float> & out, const std::vector<float> & ref) {
    double max_abs_ref = 0.0;
    double max_abs_err = 0.0;
    for (size_t r = 0; r < ref.size(); ++r) {
        max_abs_ref = std::max(max_abs_ref, (double) std::fabs(ref[r]));
        max_abs_err = std::max(max_abs_err, (double) std::fabs(out[r] - ref[r]));
    }
    return max_abs_ref > 0.0 ? max_abs_err / max_abs_ref : max_abs_err;
}

static const char * read_bw_source = MULTILINE_QUOTE(
__kernel void read_bw(__global const float4 * src, __global float * dst, const int n4) {
    float4 acc = (float4)(0.0f);
    for (int i = get_global_id(0); i < n4; i += get_global_size(0)) {
        acc += src[i];
    }
    dst[get_global_id(0)] = acc.x + acc.y + acc.z + acc.w;
}
);

// read-only access patterns of a matrix x vector product (BENCH_DMMV_PATTERN=1): nrows rows of rs bytes
// (rs multiple of 128), 8 lanes per row, 32 rows per work-group of 256, no arithmetic:
//   pat_rows:  row-major, like the kernels: step t reads 16 bytes per lane at row*rs + 128t + 16j
//   pat_tiles: tiles of 32 rows stored block-major (step t of the 32 rows of a work-group = 4 KB in a row)
static const char * pattern_source = MULTILINE_QUOTE(
__kernel void pat_rows(__global const uint4 * x, __global uint * dst, const int rs16, const int nrows) {
    const int lid = get_local_id(0);
    const int j   = lid & 7;
    const int row = get_group_id(0) * 32 + (lid >> 3);
    __global const uint4 * p = x + (size_t) min(row, nrows - 1) * rs16 + j;
    uint4 acc = (uint4)(0);
    for (int t = 0; t < rs16; t += 8) {
        acc ^= p[t];
    }
    if (row < nrows) {
        dst[row * 8 + j] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    }
}
__kernel void pat_tiles(__global const uint4 * x, __global uint * dst, const int rs16, const int nrows) {
    const int lid = get_local_id(0);
    const int row = get_group_id(0) * 32 + (lid >> 3);
    __global const uint4 * p = x + (size_t) get_group_id(0) * 32 * rs16 + lid;
    uint4 acc = (uint4)(0);
    for (int t = 0; t < rs16; t += 8) {
        acc ^= p[32 * t];
    }
    if (row < nrows) {
        dst[row * 8 + (lid & 7)] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    }
}
);

static void bench_patterns(cl_command_queue pq) {
    cl_int err;
    cl_program p = build_program_from_source(context, device, pattern_source);
    cl_kernel kr = clCreateKernel(p, "pat_rows",  &err); CL_CHECK(err);
    cl_kernel kt = clCreateKernel(p, "pat_tiles", &err); CL_CHECK(err);
    struct shape { int rs; int nrows; const char * desc; };
    const shape shapes[] = {
        { 1152, 6144, "Q4_K 2048 x 6144" },
        { 1152, 2048, "Q4_K 2048 x 2048" },
        { 3456, 2048, "Q4_K 6144 x 2048" },
        { 1664, 65536, "Q6_K-like 2048 x 65536 (rs 1664)" },
    };
    for (const shape & s : shapes) {
        const size_t bytes = (size_t) s.rs * s.nrows;
        cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, NULL, &err); CL_CHECK(err);
        cl_mem d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (size_t) s.nrows * 8 * sizeof(cl_uint), NULL, &err); CL_CHECK(err);
        {
            std::vector<uint8_t> h(bytes, 1);
            CL_CHECK(clEnqueueWriteBuffer(pq, d_x, true, 0, bytes, h.data(), 0, NULL, NULL));
        }
        const cl_int rs16 = s.rs / 16;
        const cl_int nr   = s.nrows;
        double best[2] = { 1e30, 1e30 };
        for (int round = 0; round < 5; ++round) {
            cl_kernel ks[2] = { kr, kt };
            for (int v = 0; v < 2; ++v) {
                CL_CHECK(clSetKernelArg(ks[v], 0, sizeof(cl_mem), &d_x));
                CL_CHECK(clSetKernelArg(ks[v], 1, sizeof(cl_mem), &d_d));
                CL_CHECK(clSetKernelArg(ks[v], 2, sizeof(cl_int), &rs16));
                CL_CHECK(clSetKernelArg(ks[v], 3, sizeof(cl_int), &nr));
                const size_t global = (size_t) ((s.nrows + 31) / 32) * 256;
                best[v] = std::min(best[v], time_kernel(pq, ks[v], global, 256, 10));
            }
        }
        printf("%-34s %6.1f MiB: rows %5.2f GB/s, tiles %5.2f GB/s\n", s.desc, bytes / 1048576.0,
               bytes / (best[0] * 1e-3) / 1e9, bytes / (best[1] * 1e-3) / 1e9);
        clReleaseMemObject(d_x);
        clReleaseMemObject(d_d);
    }
    clReleaseKernel(kr);
    clReleaseKernel(kt);
    clReleaseProgram(p);
}

// keeps the GPU busy for about `seconds` (the clocks of the GT 720M rise only under sustained load:
// without this, the first measurements of a run are up to 20-25% slower than the last ones)
static void gpu_warmup(cl_command_queue pq, double seconds) {
    const size_t nbytes = 64u*1024u*1024u;
    const int    n4     = (int) (nbytes / sizeof(cl_float4));
    const size_t local  = 256;
    const size_t global = 2 * 1536 * 4;
    cl_int err;
    cl_program p = build_program_from_source(context, device, read_bw_source);
    cl_kernel  k = clCreateKernel(p, "read_bw", &err);
    CL_CHECK(err);
    cl_mem d_src = clCreateBuffer(context, CL_MEM_READ_ONLY,  nbytes, NULL, &err);
    CL_CHECK(err);
    cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, global * sizeof(float), NULL, &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_src));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(int),    &n4));
    double total = 0.0;
    while (total < seconds * 1e3) {
        total += time_kernel(pq, k, global, local, 8) * 8;
    }
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
    clReleaseKernel(k);
    clReleaseProgram(p);
}

static void bench_read_bandwidth(cl_command_queue pq) {
    const size_t nbytes = 256u*1024u*1024u;
    const int    n4     = (int) (nbytes / sizeof(cl_float4));
    const size_t local  = 256;
    const size_t global = 2 * 1536 * 4; // 2 compute units, several waves of threads each

    cl_int err;
    cl_program p = build_program_from_source(context, device, read_bw_source);
    cl_kernel k = clCreateKernel(p, "read_bw", &err);
    CL_CHECK(err);

    cl_mem d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &err);
    CL_CHECK(err);
    cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, global * sizeof(float), NULL, &err);
    CL_CHECK(err);
    {
        std::vector<float> host(nbytes / sizeof(float), 1.0f);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_src, true, 0, nbytes, host.data(), 0, NULL, NULL));
    }

    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_src));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(int),    &n4));

    const double ms = time_kernel(pq, k, global, local, 5);
    printf("device read bandwidth (streaming float4 kernel, 256 MiB): %.2f GB/s (%.2f ms)\n\n", nbytes / (ms * 1e-3) / 1e9, ms);

    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
    clReleaseKernel(k);
    clReleaseProgram(p);
}

// the original (legacy) matrix x vector kernel of a K-quant type: still in the program built by ggml_cl_activate
// (generate_kernels), but the backend no longer creates it (it was used with the removed CLBlast path)
static cl_kernel * bench_old_dmmv_kernel(ggml_type type) {
    static cl_kernel kernels[GGML_TYPE_COUNT] = {};
    const char * name = type == GGML_TYPE_Q4_K ? "dequantize_mul_mat_vec_q4_K" :
                        type == GGML_TYPE_Q5_K ? "dequantize_mul_mat_vec_q5_K" :
                        type == GGML_TYPE_Q6_K ? "dequantize_mul_mat_vec_q6_K" : nullptr;
    GGML_ASSERT(name != nullptr);
    if (kernels[type] == nullptr) {
        cl_int err;
        CL_CHECK((kernels[type] = clCreateKernel(program, name, &err), err));
    }
    return &kernels[type];
}

static void bench_dmmv(cl_command_queue pq, const bench_case & bc) {
    const ggml_type type     = bc.type;
    const int64_t   ncols    = bc.ncols;
    const int64_t   nrows    = bc.nrows;
    const size_t    row_size = ggml_row_size(type, ncols);
    const size_t    q_bytes  = row_size * nrows;

    // quantize random weights chunk by chunk (no full float matrix in host memory)
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(q_bytes);
    const int64_t chunk_rows = 512;
    std::vector<float> chunk(chunk_rows * ncols);
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrows - r0);
        for (int64_t i = 0; i < nr * ncols; ++i) {
            chunk[i] = dist(rng);
        }
        ggml_quantize_chunk(type, chunk.data(), q.data() + r0 * row_size, 0, nr, ncols, nullptr);
    }
    std::vector<float> x(ncols);
    for (auto & v : x) {
        v = dist(rng);
    }

    // CPU reference: dequantize each row and dot with x in double precision
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> ref(nrows);
    {
        std::vector<float> row(ncols);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float(q.data() + r * row_size, row.data(), ncols);
            double acc = 0.0;
            for (int64_t c = 0; c < ncols; ++c) {
                acc += (double) row[c] * (double) x[c];
            }
            ref[r] = (float) acc;
        }
    }

    cl_int err;
    cl_mem d_Q = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,               NULL, &err);
    CL_CHECK(err);
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * ncols, NULL, &err);
    CL_CHECK(err);
    cl_mem d_D = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nrows, NULL, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Q, true, 0, q_bytes,               q.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Y, true, 0, sizeof(float) * ncols, x.data(), 0, NULL, NULL));

    std::vector<float> out(nrows);
    const cl_int ncols_i = (cl_int) ncols;
    const cl_int nrows_i = (cl_int) nrows;

    // original (legacy) kernel, same launch configuration as the old ggml_cl_mul_mat_q_f32 (mul_mat_vec branch)
    double old_gbs = 1.0;
    double old_err = 0.0;
    {
        cl_kernel *  dmmv   = bench_old_dmmv_kernel(type);
        const size_t local  = CL_DMMV_LOCAL_SIZE;
        const size_t global = nrows * local;
        CL_CHECK(clSetKernelArg(*dmmv, 0, sizeof(cl_mem), &d_Q));
        CL_CHECK(clSetKernelArg(*dmmv, 1, sizeof(float) * local, NULL));
        CL_CHECK(clSetKernelArg(*dmmv, 2, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(*dmmv, 3, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(*dmmv, 4, sizeof(cl_int), &ncols_i));
        const double ms = time_kernel(pq, *dmmv, global, local, bc.iters);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        old_gbs = q_bytes / (ms * 1e-3) / 1e9;
        old_err = max_rel_err(out, ref);
    }

    // fast l8 kernel, same selection and launch configuration as the backend (standard layout)
    size_t    l8_local   = 0;
    bool      l8_local_y = false;
    cl_kernel * k        = ggml_get_dmmv_l8_cl(type, ncols, &l8_local, &l8_local_y);
    GGML_ASSERT(k != nullptr);
    const size_t rows_per_wg = l8_local / 8;
    const size_t l8_global   = ((nrows + rows_per_wg - 1) / rows_per_wg) * l8_local;
    const cl_ulong zero_off  = 0;
    CL_CHECK(clSetKernelArg(*k, 0, sizeof(cl_mem),   &d_Q));
    CL_CHECK(clSetKernelArg(*k, 1, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 2, sizeof(cl_mem),   &d_Y));
    CL_CHECK(clSetKernelArg(*k, 3, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 4, sizeof(cl_mem),   &d_D));
    CL_CHECK(clSetKernelArg(*k, 5, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 6, sizeof(cl_int),   &ncols_i));
    CL_CHECK(clSetKernelArg(*k, 7, sizeof(cl_int),   &nrows_i));
    CL_CHECK(clSetKernelArg(*k, 8, sizeof(float) * l8_local, NULL));
    if (l8_local_y) {
        CL_CHECK(clSetKernelArg(*k, 9, sizeof(float) * ncols, NULL));
    }
    std::fill(out.begin(), out.end(), 0.0f);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
    const double ms     = time_kernel(pq, *k, l8_global, l8_local, bc.iters);
    const double new_gbs = q_bytes / (ms * 1e-3) / 1e9;
    CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
    const double new_err = max_rel_err(out, ref);

    printf("%-5s %5lld x %-6lld %-24s %6.1f MiB | old %5.2f GB/s (err %.0e) | l8%s W%-3zu %5.2f GB/s (err %.0e) | x%.1f",
           ggml_type_name(type), (long long) ncols, (long long) nrows, bc.desc, q_bytes / 1048576.0,
           old_gbs, old_err, l8_local_y ? "+ly" : "   ", l8_local, new_gbs, new_err, new_gbs / old_gbs);

    // A/B against the experimental v4l kernel on the same buffers
    if (ref_v4l_kernel != nullptr && type == GGML_TYPE_Q6_K && ncols <= 2048) {
        const size_t rlocal  = 256;
        const size_t rglobal = ((nrows + rlocal/8 - 1) / (rlocal/8)) * rlocal;
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 0, sizeof(cl_mem), &d_Q));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 1, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 2, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 3, sizeof(cl_int), &ncols_i));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 4, sizeof(cl_int), &nrows_i));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 5, sizeof(float) * rlocal, NULL));
        CL_CHECK(clSetKernelArg(ref_v4l_kernel, 6, sizeof(float) * ncols, NULL));
        const double rms = time_kernel(pq, ref_v4l_kernel, rglobal, rlocal, bc.iters);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        printf(" | ref v4l %5.2f GB/s (err %.0e)", q_bytes / (rms * 1e-3) / 1e9, max_rel_err(out, ref));
        // A/B/A: run the integrated kernel again after the reference, to detect order/clock effects
        const double ms2 = time_kernel(pq, *k, l8_global, l8_local, bc.iters);
        printf(" | l8 again %5.2f GB/s", q_bytes / (ms2 * 1e-3) / 1e9);
    }
    printf("\n");

    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_D);
}

static double q6x_run_l8(cl_command_queue pq, cl_mem d_Q, cl_mem d_Y, cl_mem d_D, int64_t ncols, int64_t nrows, int iters,
                         ggml_type type = GGML_TYPE_Q6_K);

static void bench_q45x(cl_command_queue pq, ggml_type type, int64_t ncols, int64_t nrows, int iters, const char * desc) {
    const size_t row_size = ggml_row_size(type, ncols);
    const size_t q_bytes  = row_size * nrows;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(q_bytes);
    const int64_t chunk_rows = 512;
    std::vector<float> chunk(chunk_rows * ncols);
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrows - r0);
        for (int64_t i = 0; i < nr * ncols; ++i) {
            chunk[i] = dist(rng);
        }
        ggml_quantize_chunk(type, chunk.data(), q.data() + r0 * row_size, 0, nr, ncols, nullptr);
    }
    std::vector<float> x(ncols);
    for (auto & v : x) {
        v = dist(rng);
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> ref(nrows);
    {
        std::vector<float> row(ncols);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float(q.data() + r * row_size, row.data(), ncols);
            double acc = 0.0;
            for (int64_t c = 0; c < ncols; ++c) {
                acc += (double) row[c] * (double) x[c];
            }
            ref[r] = (float) acc;
        }
    }
    cl_int err;
    cl_mem d_Q = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,               NULL, &err); CL_CHECK(err);
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * ncols, NULL, &err); CL_CHECK(err);
    cl_mem d_D = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nrows, NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Q, true, 0, q_bytes,               q.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Y, true, 0, sizeof(float) * ncols, x.data(), 0, NULL, NULL));
    std::vector<float> out(nrows);

    printf("%-5s %5lld x %-6lld %-24s %6.1f MiB\n", ggml_type_name(type), (long long) ncols, (long long) nrows, desc, q_bytes / 1048576.0);
    q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters, type);
    const double l8_ms = q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters, type);
    CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
    printf("  integrated l8                 %7.3f ms %5.2f GB/s (err %.0e)\n", l8_ms, q_bytes / (l8_ms * 1e-3) / 1e9, max_rel_err(out, ref));

    cl_ulong local_mem = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL));
    const cl_int ncols_i = (cl_int) ncols;
    const cl_int nrows_i = (cl_int) nrows;
    const bool   regs    = getenv("BENCH_DMMV_Q45X_REGS") != nullptr;
    for (const q45x_variant & v : q45x_variants) {
        const int64_t ych  = std::min<int64_t>(v.ych, ncols);
        const size_t  lmem = sizeof(float) * (v.lwg + ych + ych / 8);
        char name[64];
        snprintf(name, sizeof(name), "q45x LWG %3d YV4 %d YCH %4d", v.lwg, v.yv4, v.ych);
        if (lmem > local_mem) {
            printf("  %s: skipped\n", name);
            continue;
        }
        const std::string src = "#define LWG " + std::to_string(v.lwg) + "\n#define YV4 " + std::to_string(v.yv4) +
                                "\n#define YCH " + std::to_string(v.ych) + "\n#define BS " + std::to_string(ggml_type_size(type)) + "\n" + q45x_source;
        if (regs) {
            const char * src_ptr = src.c_str();
            cl_program p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
            CL_CHECK(err);
            const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -DQK_K=256 -cl-nv-verbose";
            CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
            size_t log_size = 0;
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
            std::string log(log_size, '\0');
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL));
            const size_t ur = log.find("Used");
            printf("  %s: %s\n", name, ur == std::string::npos ? "?" : log.substr(ur, log.find('\n', ur) - ur).c_str());
            // BENCH_DMMV_Q45X_PTX=<prefix>: also the PTX of every variant
            if (const char * prefix = getenv("BENCH_DMMV_Q45X_PTX")) {
                size_t bin_size = 0;
                CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL));
                std::vector<unsigned char> bin(bin_size);
                unsigned char * bin_ptr = bin.data();
                CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, NULL));
                const std::string path = std::string(prefix) + "-" + ggml_type_name(type) + "-" + std::to_string(v.lwg) + "-" +
                                         std::to_string(v.yv4) + ".ptx";
                FILE * f = fopen(path.c_str(), "wb");
                if (f) {
                    fwrite(bin.data(), 1, bin.size(), f);
                    fclose(f);
                }
            }
            clReleaseProgram(p);
            continue;
        }
        cl_program p = build_program_from_source(context, device, src.c_str());
        cl_kernel  k = clCreateKernel(p, "q45x", &err);
        CL_CHECK(err);
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_Q));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_int), &ncols_i));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int), &nrows_i));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(float) * v.lwg, NULL));
        CL_CHECK(clSetKernelArg(k, 6, sizeof(float) * ych, NULL));
        CL_CHECK(clSetKernelArg(k, 7, sizeof(float) * ych / 8, NULL));
        const size_t rows_per_wg = v.lwg / 8;
        const size_t global      = ((nrows + rows_per_wg - 1) / rows_per_wg) * v.lwg;
        std::fill(out.begin(), out.end(), 0.0f);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        const double ms = time_kernel(pq, k, global, v.lwg, iters);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        printf("  %s %7.3f ms %5.2f GB/s (err %.0e) x%.2f\n", name, ms, q_bytes / (ms * 1e-3) / 1e9, max_rel_err(out, ref), l8_ms / ms);
        clReleaseKernel(k);
        clReleaseProgram(p);
    }
    if (!regs && type == GGML_TYPE_Q4_K) {
        // the same matrix repacked on the host (q4r layout)
        const int nb = (int) (ncols / 256);
        std::vector<uint8_t> qr(q_bytes);
        for (int64_t r = 0; r < nrows; ++r) {
            q4r_repack_row(q.data() + r * row_size, qr.data() + r * row_size, nb);
        }
        cl_mem d_QR = clCreateBuffer(context, CL_MEM_READ_ONLY, q_bytes, NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_QR, true, 0, q_bytes, qr.data(), 0, NULL, NULL));
        // BENCH_DMMV_Q4R_OPTS: extra build options of the q4r program, several sets separated by ';'
        std::vector<std::string> extra_opts = { "" };
        if (const char * eo = getenv("BENCH_DMMV_Q4R_OPTS")) {
            std::string s = eo;
            size_t at = 0;
            while (true) {
                const size_t sc = s.find(';', at);
                extra_opts.push_back(s.substr(at, sc == std::string::npos ? std::string::npos : sc - at));
                if (sc == std::string::npos) {
                    break;
                }
                at = sc + 1;
            }
        }
        for (const std::string & xo : extra_opts)
        for (int qv : { 4, 2, 1 })
        for (int lwg : { 128, 256 }) {
            const int64_t ych = std::min<int64_t>(2048, ncols);
            const std::string src = "#define LWG " + std::to_string(lwg) + "\n#define YCH 2048\n#define QV " + std::to_string(qv) + "\n" + q4r_source;
            const char * src_ptr = src.c_str();
            cl_program p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
            CL_CHECK(err);
            const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -DQK_K=256 " + xo;
            CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
            cl_kernel  k = clCreateKernel(p, "q4r", &err);
            CL_CHECK(err);
            CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_QR));
            CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_Y));
            CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &d_D));
            CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_int), &ncols_i));
            CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int), &nrows_i));
            CL_CHECK(clSetKernelArg(k, 5, sizeof(float) * lwg, NULL));
            CL_CHECK(clSetKernelArg(k, 6, sizeof(float) * (ych + ych / 8), NULL));
            CL_CHECK(clSetKernelArg(k, 7, sizeof(float) * ych / 8, NULL));
            const size_t rows_per_wg = lwg / 8;
            const size_t global      = ((nrows + rows_per_wg - 1) / rows_per_wg) * lwg;
            std::fill(out.begin(), out.end(), 0.0f);
            CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
            const double ms = time_kernel(pq, k, global, lwg, iters);
            CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
            printf("  q4r LWG %3d QV %d %-21s %7.3f ms %5.2f GB/s (err %.0e) x%.2f\n", lwg, qv, xo.c_str(), ms, q_bytes / (ms * 1e-3) / 1e9,
                   max_rel_err(out, ref), l8_ms / ms);
            clReleaseKernel(k);
            clReleaseProgram(p);
        }
        // q4rt: several groups of rows per work-group (ncols <= 2048)
        if (ncols <= 2048) {
            const int lwg = 256;
            const std::string src = "#define LWG 256\n#define YCH 2048\n#define QV 2\n" + std::string(q4r_source) + q4rt_source;
            cl_program p = build_program_from_source(context, device, src.c_str());
            cl_kernel  k = clCreateKernel(p, "q4rt", &err);
            CL_CHECK(err);
            for (int T : { 1, 2, 4, 8 }) {
                const cl_int T_i = T;
                CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_QR));
                CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_Y));
                CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &d_D));
                CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_int), &ncols_i));
                CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int), &nrows_i));
                CL_CHECK(clSetKernelArg(k, 5, sizeof(float) * lwg, NULL));
                CL_CHECK(clSetKernelArg(k, 6, sizeof(float) * (ncols + ncols / 8), NULL));
                CL_CHECK(clSetKernelArg(k, 7, sizeof(float) * ncols / 8, NULL));
                CL_CHECK(clSetKernelArg(k, 8, sizeof(cl_int), &T_i));
                const size_t rows_per_wg = lwg / 8 * T;
                const size_t global      = ((nrows + rows_per_wg - 1) / rows_per_wg) * lwg;
                std::fill(out.begin(), out.end(), 0.0f);
                CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
                const double ms = time_kernel(pq, k, global, lwg, iters);
                CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
                printf("  q4rt LWG 256 T %d                        %7.3f ms %5.2f GB/s (err %.0e) x%.2f\n", T, ms, q_bytes / (ms * 1e-3) / 1e9,
                       max_rel_err(out, ref), l8_ms / ms);
            }
            clReleaseKernel(k);
            clReleaseProgram(p);
        }
        clReleaseMemObject(d_QR);
    }
    if (!regs) {
        const double l8_ms2 = q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters, type);
        printf("  integrated l8 again           %7.3f ms %5.2f GB/s\n", l8_ms2, q_bytes / (l8_ms2 * 1e-3) / 1e9);
    }
    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_D);
}

struct ab_cand {
    std::string name;
    cl_kernel   k;
    size_t      global;
    size_t      local;
    cl_mem      w;
    double      best = 1e30;
    double      err  = 0.0;
};

// q6rt: dmmv_q6r of the backend (q6r layout) where every work-group handles T groups of rows with y staged
// once (ncols <= 2048, one chunk)
static const char * q6rt_source = R"CLC(
__kernel void q6rt(__global const uchar * xq, __global const float * y, __global float * dst,
                   const int ncols, const int nrows, __local float * tmp, __local float4 * yl4, __local float4 * ysl4, const int T) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int j   = lid & 7;
    const int h   = j >> 2;
    const int p   = j & 3;
    const int nb  = ncols / QK_K;
    __local float * yl  = (__local float *) yl4;
    __local float * ysl = (__local float *) ysl4;
    for (int e = lid; e < ncols/8; e += lsz) {
        const float4 u = vload4(0, y + 8*e);
        const float4 w = vload4(0, y + 8*e + 4);
        vstore4(u, 0, yl + Q6R_YOFF(8*e));
        vstore4(w, 0, yl + Q6R_YOFF(8*e) + 4);
        ysl[(e & ~15) | ((e & 3) << 2) | ((e >> 2) & 3)] = ((u.x + u.y) + (u.z + u.w)) + ((w.x + w.y) + (w.z + w.w));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int rg = 0; rg < T; ++rg) {
        const int row = (get_group_id(0) * T + rg) * (lsz >> 3) + (lid >> 3);
        __global const uchar * rb = xq + (size_t) min(row, nrows - 1) * nb * 210;
        __global const uchar * pq = rb + 8*j;
        __global const uchar * ph = rb + (size_t) nb*128 + 8*j;
        __global const uint  * ps = (__global const uint *) (rb + (size_t) nb*192) + 2*h + (p >> 1);
        __global const half  * pd = (__global const half *) (rb + (size_t) nb*208);
        float acc = 0.0f;
        for (int i = 0; i < nb; ++i) {
            const uint2 a = ((__global const uint2 *) pq)[0];
            const uint2 b = ((__global const uint2 *) (pq + 64))[0];
            const uint2 c = ((__global const uint2 *) ph)[0];
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
        tmp[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int st = 4; st > 0; st >>= 1) {
            if (j < st) {
                tmp[lid] += tmp[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (j == 0 && row < nrows) {
            dst[row] = tmp[lid];
        }
    }
}
)CLC";

// interleaved A/B of the q6r matrix x vector kernel of the backend against q6rt (BENCH_DMMV_Q6AB=1)
static void bench_q6_ab(cl_command_queue pq, int64_t ncols, int64_t nrows, int iters, int rounds, const char * desc) {
    const ggml_type type     = GGML_TYPE_Q6_K;
    const size_t    row_size = ggml_row_size(type, ncols);
    const size_t    q_bytes  = row_size * nrows;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(q_bytes);
    std::vector<float> chunk(512 * ncols);
    for (int64_t r0 = 0; r0 < nrows; r0 += 512) {
        const int64_t nr = std::min<int64_t>(512, nrows - r0);
        for (int64_t i = 0; i < nr * ncols; ++i) {
            chunk[i] = dist(rng);
        }
        ggml_quantize_chunk(type, chunk.data(), q.data() + r0 * row_size, 0, nr, ncols, nullptr);
    }
    std::vector<float> x(ncols);
    for (auto & v : x) {
        v = dist(rng);
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> ref(nrows);
    {
        std::vector<float> row(ncols);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float(q.data() + r * row_size, row.data(), ncols);
            double acc = 0.0;
            for (int64_t c = 0; c < ncols; ++c) {
                acc += (double) row[c] * (double) x[c];
            }
            ref[r] = (float) acc;
        }
    }
    const int nb = (int) (ncols / 256);
    std::vector<uint8_t> qr(q_bytes);
    for (int64_t r = 0; r < nrows; ++r) {
        ggml_cl_q6r_pack_row(q.data() + r * row_size, qr.data() + r * row_size, nb);
    }
    cl_int err;
    cl_mem d_QR = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,               NULL, &err); CL_CHECK(err);
    cl_mem d_Y  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * ncols, NULL, &err); CL_CHECK(err);
    cl_mem d_D  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nrows, NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_QR, true, 0, q_bytes,               qr.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Y,  true, 0, sizeof(float) * ncols, x.data(),  0, NULL, NULL));
    const cl_int   ncols_i  = (cl_int) ncols;
    const cl_int   nrows_i  = (cl_int) nrows;
    const cl_ulong zero_off = 0;
    std::vector<ab_cand> cands;
    {
        ab_cand c;
        c.k = be_k.dmmv_q6r_v2;
        const cl_int ych = be_q6r_ych;
        CL_CHECK(clSetKernelArg(c.k, 0, sizeof(cl_mem),   &d_QR));
        CL_CHECK(clSetKernelArg(c.k, 1, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(c.k, 2, sizeof(cl_mem),   &d_Y));
        CL_CHECK(clSetKernelArg(c.k, 3, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(c.k, 4, sizeof(cl_mem),   &d_D));
        CL_CHECK(clSetKernelArg(c.k, 5, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(c.k, 6, sizeof(cl_int),   &ncols_i));
        CL_CHECK(clSetKernelArg(c.k, 7, sizeof(cl_int),   &nrows_i));
        CL_CHECK(clSetKernelArg(c.k, 8, sizeof(cl_int),   &ych));
        CL_CHECK(clSetKernelArg(c.k, 9,  sizeof(float) * be_q6r_wg, NULL));
        CL_CHECK(clSetKernelArg(c.k, 10, sizeof(float) * (ych + ych/32), NULL));
        CL_CHECK(clSetKernelArg(c.k, 11, sizeof(float) * (ych/8), NULL));
        c.name   = "backend dmmv_q6r_v2";
        c.local  = be_q6r_wg;
        c.global = ((nrows + be_q6r_wg/8 - 1) / (be_q6r_wg/8)) * be_q6r_wg;
        cands.push_back(c);
    }
    const std::string src = std::string(be_q6r_common) + be_q6r_source + q6rt_source;
    cl_program p = build_program_from_source(context, device, src.c_str());
    for (int lwg : { 256, 128 }) {
        for (int T : { 1, 2, 4, 8 }) {
            ab_cand c;
            c.k = clCreateKernel(p, "q6rt", &err);
            CL_CHECK(err);
            const cl_int T_i = T;
            CL_CHECK(clSetKernelArg(c.k, 0, sizeof(cl_mem), &d_QR));
            CL_CHECK(clSetKernelArg(c.k, 1, sizeof(cl_mem), &d_Y));
            CL_CHECK(clSetKernelArg(c.k, 2, sizeof(cl_mem), &d_D));
            CL_CHECK(clSetKernelArg(c.k, 3, sizeof(cl_int), &ncols_i));
            CL_CHECK(clSetKernelArg(c.k, 4, sizeof(cl_int), &nrows_i));
            CL_CHECK(clSetKernelArg(c.k, 5, sizeof(float) * lwg, NULL));
            CL_CHECK(clSetKernelArg(c.k, 6, sizeof(float) * (ncols + ncols/32), NULL));
            CL_CHECK(clSetKernelArg(c.k, 7, sizeof(float) * (ncols/8), NULL));
            CL_CHECK(clSetKernelArg(c.k, 8, sizeof(cl_int), &T_i));
            c.name   = "q6rt LWG " + std::to_string(lwg) + " T " + std::to_string(T);
            c.local  = lwg;
            c.global = ((nrows + lwg/8*T - 1) / (lwg/8*T)) * lwg;
            cands.push_back(c);
        }
    }
    std::vector<float> out(nrows);
    for (ab_cand & c : cands) {
        std::fill(out.begin(), out.end(), 0.0f);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        time_kernel(pq, c.k, c.global, c.local, 1);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        c.err = max_rel_err(out, ref);
    }
    for (int r = 0; r < rounds; ++r) {
        for (ab_cand & c : cands) {
            c.best = std::min(c.best, time_kernel(pq, c.k, c.global, c.local, iters));
        }
    }
    printf("q6_K %5lld x %-6lld %-24s %6.1f MiB (best of %d rounds of %d)\n", (long long) ncols, (long long) nrows, desc,
           q_bytes / 1048576.0, rounds, iters);
    for (const ab_cand & c : cands) {
        printf("  %-20s %7.3f ms %5.2f GB/s (err %.0e) x%.2f\n", c.name.c_str(), c.best, q_bytes / (c.best * 1e-3) / 1e9, c.err,
               cands[0].best / c.best);
    }
    for (size_t i = 1; i < cands.size(); ++i) {
        clReleaseKernel(cands[i].k);
    }
    clReleaseProgram(p);
    clReleaseMemObject(d_QR);
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_D);
}

// interleaved A/B of Q4_K matrix x vector kernels (BENCH_DMMV_Q4AB=1): every candidate is prepared once
// (its own kernel object, arguments set), then all of them are timed in `rounds` rounds and the best
// time of every candidate is kept, so that clock changes affect all of them alike
static void bench_q4_ab(cl_command_queue pq, int64_t ncols, int64_t nrows, int iters, int rounds, const char * desc) {
    const ggml_type type     = GGML_TYPE_Q4_K;
    const size_t    row_size = ggml_row_size(type, ncols);
    const size_t    q_bytes  = row_size * nrows;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(q_bytes);
    std::vector<float> chunk(512 * ncols);
    for (int64_t r0 = 0; r0 < nrows; r0 += 512) {
        const int64_t nr = std::min<int64_t>(512, nrows - r0);
        for (int64_t i = 0; i < nr * ncols; ++i) {
            chunk[i] = dist(rng);
        }
        ggml_quantize_chunk(type, chunk.data(), q.data() + r0 * row_size, 0, nr, ncols, nullptr);
    }
    std::vector<float> x(ncols);
    for (auto & v : x) {
        v = dist(rng);
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> ref(nrows);
    {
        std::vector<float> row(ncols);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float(q.data() + r * row_size, row.data(), ncols);
            double acc = 0.0;
            for (int64_t c = 0; c < ncols; ++c) {
                acc += (double) row[c] * (double) x[c];
            }
            ref[r] = (float) acc;
        }
    }
    const int nb = (int) (ncols / 256);
    std::vector<uint8_t> qr(q_bytes);
    for (int64_t r = 0; r < nrows; ++r) {
        q4r_repack_row(q.data() + r * row_size, qr.data() + r * row_size, nb);
    }
    cl_int err;
    cl_mem d_Q  = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,               NULL, &err); CL_CHECK(err);
    cl_mem d_QR = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,               NULL, &err); CL_CHECK(err);
    cl_mem d_Y  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * ncols, NULL, &err); CL_CHECK(err);
    cl_mem d_D  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nrows, NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Q,  true, 0, q_bytes,               q.data(),  0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_QR, true, 0, q_bytes,               qr.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Y,  true, 0, sizeof(float) * ncols, x.data(),  0, NULL, NULL));
    const cl_int   ncols_i  = (cl_int) ncols;
    const cl_int   nrows_i  = (cl_int) nrows;
    const cl_ulong zero_off = 0;
    const int64_t  ych      = std::min<int64_t>(2048, ncols);
    std::vector<ab_cand> cands;
    std::vector<cl_program> progs;

    // integrated l8
    {
        size_t l8_local = 0;
        bool   l8_ly    = false;
        cl_kernel * k = ggml_get_dmmv_l8_cl(type, ncols, &l8_local, &l8_ly);
        CL_CHECK(clSetKernelArg(*k, 0, sizeof(cl_mem),   &d_Q));
        CL_CHECK(clSetKernelArg(*k, 1, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(*k, 2, sizeof(cl_mem),   &d_Y));
        CL_CHECK(clSetKernelArg(*k, 3, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(*k, 4, sizeof(cl_mem),   &d_D));
        CL_CHECK(clSetKernelArg(*k, 5, sizeof(cl_ulong), &zero_off));
        CL_CHECK(clSetKernelArg(*k, 6, sizeof(cl_int),   &ncols_i));
        CL_CHECK(clSetKernelArg(*k, 7, sizeof(cl_int),   &nrows_i));
        CL_CHECK(clSetKernelArg(*k, 8, sizeof(float) * l8_local, NULL));
        ab_cand c;
        c.name   = "integrated l8";
        c.k      = *k;
        c.local  = l8_local;
        c.global = ((nrows + l8_local/8 - 1) / (l8_local/8)) * l8_local;
        c.w      = d_Q;
        cands.push_back(c);
    }
    // q45x (standard layout, y in local memory)
    for (int lwg : { 256 }) {
        const std::string src = "#define LWG " + std::to_string(lwg) + "\n#define YV4 0\n#define YCH 2048\n#define BS 144\n" + q45x_source +
                                q45xt_source;
        cl_program p = build_program_from_source(context, device, src.c_str());
        progs.push_back(p);
        if (ncols <= 2048) {
            for (int T : { 2, 4 }) {
                ab_cand ct;
                ct.k = clCreateKernel(p, "q45xt", &err);
                CL_CHECK(err);
                const cl_int T_i = T;
                CL_CHECK(clSetKernelArg(ct.k, 0, sizeof(cl_mem), &d_Q));
                CL_CHECK(clSetKernelArg(ct.k, 1, sizeof(cl_mem), &d_Y));
                CL_CHECK(clSetKernelArg(ct.k, 2, sizeof(cl_mem), &d_D));
                CL_CHECK(clSetKernelArg(ct.k, 3, sizeof(cl_int), &ncols_i));
                CL_CHECK(clSetKernelArg(ct.k, 4, sizeof(cl_int), &nrows_i));
                CL_CHECK(clSetKernelArg(ct.k, 5, sizeof(float) * lwg, NULL));
                CL_CHECK(clSetKernelArg(ct.k, 6, sizeof(float) * ncols, NULL));
                CL_CHECK(clSetKernelArg(ct.k, 7, sizeof(float) * ncols / 8, NULL));
                CL_CHECK(clSetKernelArg(ct.k, 8, sizeof(cl_int), &T_i));
                ct.name   = "q45xt T " + std::to_string(T);
                ct.local  = lwg;
                ct.global = ((nrows + lwg/8*T - 1) / (lwg/8*T)) * lwg;
                ct.w      = d_Q;
                cands.push_back(ct);
            }
        }
        ab_cand c;
        c.k = clCreateKernel(p, "q45x", &err);
        CL_CHECK(err);
        CL_CHECK(clSetKernelArg(c.k, 0, sizeof(cl_mem), &d_Q));
        CL_CHECK(clSetKernelArg(c.k, 1, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(c.k, 2, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(c.k, 3, sizeof(cl_int), &ncols_i));
        CL_CHECK(clSetKernelArg(c.k, 4, sizeof(cl_int), &nrows_i));
        CL_CHECK(clSetKernelArg(c.k, 5, sizeof(float) * lwg, NULL));
        CL_CHECK(clSetKernelArg(c.k, 6, sizeof(float) * ych, NULL));
        CL_CHECK(clSetKernelArg(c.k, 7, sizeof(float) * ych / 8, NULL));
        c.name   = "q45x LWG " + std::to_string(lwg);
        c.local  = lwg;
        c.global = ((nrows + lwg/8 - 1) / (lwg/8)) * lwg;
        c.w      = d_Q;
        cands.push_back(c);
    }
    // q4r QV 2 and QV 4
    for (int qv : { 2, 4 }) {
        const int lwg = 256;
        const std::string src = "#define LWG 256\n#define YCH 2048\n#define QV " + std::to_string(qv) + "\n" + std::string(q4r_source) +
                                (ncols <= 2048 ? q4rt_source : "");
        cl_program p = build_program_from_source(context, device, src.c_str());
        progs.push_back(p);
        ab_cand c;
        c.k = clCreateKernel(p, "q4r", &err);
        CL_CHECK(err);
        CL_CHECK(clSetKernelArg(c.k, 0, sizeof(cl_mem), &d_QR));
        CL_CHECK(clSetKernelArg(c.k, 1, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(c.k, 2, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(c.k, 3, sizeof(cl_int), &ncols_i));
        CL_CHECK(clSetKernelArg(c.k, 4, sizeof(cl_int), &nrows_i));
        CL_CHECK(clSetKernelArg(c.k, 5, sizeof(float) * lwg, NULL));
        CL_CHECK(clSetKernelArg(c.k, 6, sizeof(float) * (ych + ych / 8), NULL));
        CL_CHECK(clSetKernelArg(c.k, 7, sizeof(float) * ych / 8, NULL));
        c.name   = "q4r QV " + std::to_string(qv);
        c.local  = lwg;
        c.global = ((nrows + lwg/8 - 1) / (lwg/8)) * lwg;
        c.w      = d_QR;
        cands.push_back(c);
        if (qv == 2 && ncols <= 2048) {
            for (int T : { 2, 4 }) {
                ab_cand ct;
                ct.k = clCreateKernel(p, "q4rt", &err);
                CL_CHECK(err);
                const cl_int T_i = T;
                CL_CHECK(clSetKernelArg(ct.k, 0, sizeof(cl_mem), &d_QR));
                CL_CHECK(clSetKernelArg(ct.k, 1, sizeof(cl_mem), &d_Y));
                CL_CHECK(clSetKernelArg(ct.k, 2, sizeof(cl_mem), &d_D));
                CL_CHECK(clSetKernelArg(ct.k, 3, sizeof(cl_int), &ncols_i));
                CL_CHECK(clSetKernelArg(ct.k, 4, sizeof(cl_int), &nrows_i));
                CL_CHECK(clSetKernelArg(ct.k, 5, sizeof(float) * lwg, NULL));
                CL_CHECK(clSetKernelArg(ct.k, 6, sizeof(float) * (ncols + ncols / 8), NULL));
                CL_CHECK(clSetKernelArg(ct.k, 7, sizeof(float) * ncols / 8, NULL));
                CL_CHECK(clSetKernelArg(ct.k, 8, sizeof(cl_int), &T_i));
                ct.name   = "q4rt QV 2 T " + std::to_string(T);
                ct.local  = lwg;
                ct.global = ((nrows + lwg/8*T - 1) / (lwg/8*T)) * lwg;
                ct.w      = d_QR;
                cands.push_back(ct);
            }
        }
    }

    std::vector<float> out(nrows);
    for (ab_cand & c : cands) {
        std::fill(out.begin(), out.end(), 0.0f);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        time_kernel(pq, c.k, c.global, c.local, 1);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        c.err = max_rel_err(out, ref);
    }
    for (int r = 0; r < rounds; ++r) {
        for (ab_cand & c : cands) {
            c.best = std::min(c.best, time_kernel(pq, c.k, c.global, c.local, iters));
        }
    }
    printf("q4_K %5lld x %-6lld %-24s %6.1f MiB (best of %d rounds of %d)\n", (long long) ncols, (long long) nrows, desc,
           q_bytes / 1048576.0, rounds, iters);
    for (const ab_cand & c : cands) {
        printf("  %-20s %7.3f ms %5.2f GB/s (err %.0e) x%.2f\n", c.name.c_str(), c.best, q_bytes / (c.best * 1e-3) / 1e9, c.err,
               cands[0].best / c.best);
    }
    for (size_t i = 1; i < cands.size(); ++i) {
        clReleaseKernel(cands[i].k);
    }
    for (cl_program p : progs) {
        clReleaseProgram(p);
    }
    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_QR);
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_D);
}

// runs the integrated l8 kernel with the configuration of the backend; returns the kernel time in ms
static double q6x_run_l8(cl_command_queue pq, cl_mem d_Q, cl_mem d_Y, cl_mem d_D, int64_t ncols, int64_t nrows, int iters,
                         ggml_type type) {
    size_t    l8_local   = 0;
    bool      l8_local_y = false;
    cl_kernel * k        = ggml_get_dmmv_l8_cl(type, ncols, &l8_local, &l8_local_y);
    const size_t   rows_per_wg = l8_local / 8;
    const size_t   l8_global   = ((nrows + rows_per_wg - 1) / rows_per_wg) * l8_local;
    const cl_ulong zero_off    = 0;
    const cl_int   ncols_i     = (cl_int) ncols;
    const cl_int   nrows_i     = (cl_int) nrows;
    CL_CHECK(clSetKernelArg(*k, 0, sizeof(cl_mem),   &d_Q));
    CL_CHECK(clSetKernelArg(*k, 1, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 2, sizeof(cl_mem),   &d_Y));
    CL_CHECK(clSetKernelArg(*k, 3, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 4, sizeof(cl_mem),   &d_D));
    CL_CHECK(clSetKernelArg(*k, 5, sizeof(cl_ulong), &zero_off));
    CL_CHECK(clSetKernelArg(*k, 6, sizeof(cl_int),   &ncols_i));
    CL_CHECK(clSetKernelArg(*k, 7, sizeof(cl_int),   &nrows_i));
    CL_CHECK(clSetKernelArg(*k, 8, sizeof(float) * l8_local, NULL));
    if (l8_local_y) {
        CL_CHECK(clSetKernelArg(*k, 9, sizeof(float) * ncols, NULL));
    }
    return time_kernel(pq, *k, l8_global, l8_local, iters);
}

static void bench_q6x(cl_command_queue pq, int64_t ncols, int64_t nrows, int iters, const char * desc) {
    const ggml_type type     = GGML_TYPE_Q6_K;
    const size_t    row_size = ggml_row_size(type, ncols);
    const size_t    q_bytes  = row_size * nrows;
    GGML_ASSERT((ncols / 256) % 2 == 0);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> q(q_bytes);
    const int64_t chunk_rows = 512;
    std::vector<float> chunk(chunk_rows * ncols);
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrows - r0);
        for (int64_t i = 0; i < nr * ncols; ++i) {
            chunk[i] = dist(rng);
        }
        ggml_quantize_chunk(type, chunk.data(), q.data() + r0 * row_size, 0, nr, ncols, nullptr);
    }
    std::vector<float> x(ncols);
    for (auto & v : x) {
        v = dist(rng);
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> ref(nrows);
    {
        std::vector<float> row(ncols);
        for (int64_t r = 0; r < nrows; ++r) {
            traits->to_float(q.data() + r * row_size, row.data(), ncols);
            double acc = 0.0;
            for (int64_t c = 0; c < ncols; ++c) {
                acc += (double) row[c] * (double) x[c];
            }
            ref[r] = (float) acc;
        }
    }

    cl_int err;
    cl_mem d_Q  = clCreateBuffer(context, CL_MEM_READ_ONLY,  q_bytes,                   NULL, &err); CL_CHECK(err);
    cl_mem d_Y  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * ncols,     NULL, &err); CL_CHECK(err);
    cl_mem d_YS = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ncols / 8, NULL, &err); CL_CHECK(err);
    cl_mem d_D  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nrows,     NULL, &err); CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Q, true, 0, q_bytes,               q.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(pq, d_Y, true, 0, sizeof(float) * ncols, x.data(), 0, NULL, NULL));
    std::vector<float> out(nrows);

    printf("Q6_K %5lld x %-6lld %-24s %6.1f MiB\n", (long long) ncols, (long long) nrows, desc, q_bytes / 1048576.0);
    q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters); // warm-up (clocks, first use of the buffers)
    const double l8_ms = q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters);
    CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
    printf("  integrated l8                     %6.3f ms %5.2f GB/s (err %.0e)\n", l8_ms, q_bytes / (l8_ms * 1e-3) / 1e9, max_rel_err(out, ref));

    cl_ulong local_mem = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL));
    const cl_int ncols_i = (cl_int) ncols;
    const cl_int nrows_i = (cl_int) nrows;
    const cl_int n8_i    = (cl_int) (ncols / 8);
    const bool   regs    = getenv("BENCH_DMMV_Q6X_REGS") != nullptr;
    for (const q6x_variant & v : q6x_variants) {
        if (!regs && getenv("BENCH_DMMV_Q6R_ONLY")) {
            break;
        }
        const size_t yl_floats = ncols + ncols / 32;
        const size_t lmem      = sizeof(float) * (v.r * v.lwg + (v.yl ? yl_floats + ncols / 8 : 0));
        if (lmem > local_mem) {
            printf("  variant LWG %3d R %d YL %d YV4 %d ODDLD %d: needs %zu bytes of local memory, skipped\n", v.lwg, v.r, v.yl, v.yv4, v.oddld, lmem);
            continue;
        }
        const std::string src = q6x_defines(v) + q6x_source;
        cl_program p;
        if (regs) {
            const char * src_ptr = src.c_str();
            p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
            CL_CHECK(err);
            const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -DQK_K=256 -cl-nv-verbose";
            CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
            size_t log_size = 0;
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
            std::string log(log_size, '\0');
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL));
            // keep only the ptxas lines of kernel q6x
            const size_t at = log.find("Compiling entry function 'q6x'");
            printf("  variant LWG %3d R %d YL %d YV4 %d ODDLD %d: %s\n", v.lwg, v.r, v.yl, v.yv4, v.oddld,
                   at == std::string::npos ? log.c_str() : log.substr(at, 400).c_str());
            // BENCH_DMMV_Q6X_PTX=<prefix>: also the program binary (PTX on NVIDIA) of every variant
            if (const char * prefix = getenv("BENCH_DMMV_Q6X_PTX")) {
                size_t bin_size = 0;
                CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL));
                std::vector<unsigned char> bin(bin_size);
                unsigned char * bin_ptr = bin.data();
                CL_CHECK(clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, NULL));
                const std::string path = std::string(prefix) + "-" + std::to_string(v.lwg) + "-" + std::to_string(v.r) + "-" +
                                         std::to_string(v.yl) + std::to_string(v.yv4) + std::to_string(v.oddld) + ".ptx";
                FILE * f = fopen(path.c_str(), "wb");
                if (f) {
                    fwrite(bin.data(), 1, bin.size(), f);
                    fclose(f);
                }
            }
            clReleaseProgram(p);
            continue;
        }
        p = build_program_from_source(context, device, src.c_str());
        cl_kernel ks = clCreateKernel(p, "q6x_ysum8", &err); CL_CHECK(err);
        cl_kernel k  = clCreateKernel(p, "q6x",       &err); CL_CHECK(err);

        CL_CHECK(clSetKernelArg(ks, 0, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(ks, 1, sizeof(cl_mem), &d_YS));
        CL_CHECK(clSetKernelArg(ks, 2, sizeof(cl_int), &n8_i));
        const size_t ys_local  = 64;
        const size_t ys_global = ((ncols / 8 + ys_local - 1) / ys_local) * ys_local;
        const double ys_ms = time_kernel(pq, ks, ys_global, ys_local, iters);

        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_Q));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_Y));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &d_YS));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &d_D));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int), &ncols_i));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_int), &nrows_i));
        CL_CHECK(clSetKernelArg(k, 6, sizeof(float) * v.r * v.lwg, NULL));
        if (v.yl) {
            CL_CHECK(clSetKernelArg(k, 7, sizeof(float) * yl_floats, NULL));
            CL_CHECK(clSetKernelArg(k, 8, sizeof(float) * ncols / 8, NULL));
        }
        const size_t rows_per_wg = v.lwg / 8 * v.r;
        const size_t global      = ((nrows + rows_per_wg - 1) / rows_per_wg) * v.lwg;
        std::fill(out.begin(), out.end(), 0.0f);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        const double ms = time_kernel(pq, k, global, v.lwg, iters);
        CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
        printf("  variant LWG %3d R %d YL %d YV4 %d ODDLD %d %6.3f ms %5.2f GB/s (err %.0e) | +ysum %.3f ms: %5.2f GB/s, x%.2f\n",
               v.lwg, v.r, v.yl, v.yv4, v.oddld, ms, q_bytes / (ms * 1e-3) / 1e9, max_rel_err(out, ref),
               ys_ms, q_bytes / ((ms + ys_ms) * 1e-3) / 1e9, l8_ms / (ms + ys_ms));
        clReleaseKernel(k);
        clReleaseKernel(ks);
        clReleaseProgram(p);
    }
    if (!regs && getenv("BENCH_DMMV_Q6R")) {
        // the same matrix repacked on the host (q6r layout)
        const int nb = (int) (ncols / 256);
        std::vector<uint8_t> qr(q_bytes);
        for (int64_t r = 0; r < nrows; ++r) {
            q6r_repack_row(q.data() + r * row_size, qr.data() + r * row_size, nb);
        }
        cl_mem d_QR = clCreateBuffer(context, CL_MEM_READ_ONLY, q_bytes, NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(pq, d_QR, true, 0, q_bytes, qr.data(), 0, NULL, NULL));
        const bool rregs = getenv("BENCH_DMMV_Q6R_REGS") != nullptr;
        for (const q6r_variant & v : q6r_variants) {
            const int64_t ych       = v.ych > 0 ? std::min<int64_t>(v.ych, ncols) : ncols;
            const size_t  yl_floats = ych + ych / 32;
            const size_t  lmem      = sizeof(float) * (v.r * v.lwg + yl_floats + ych / 8);
            char name[96];
            snprintf(name, sizeof(name), "q6r LWG %3d V2 %d YCH %4d R %d UNR %d", v.lwg, v.v2, v.ych, v.r, v.unr);
            if (lmem > local_mem || (v.v2 && row_size % 8 != 0)) {
                printf("  %s: skipped\n", name);
                continue;
            }
            const std::string src = "#define LWG " + std::to_string(v.lwg) + "\n#define V2 " + std::to_string(v.v2) +
                                    "\n#define YCH " + std::to_string(v.ych) + "\n#define R " + std::to_string(v.r) +
                                    "\n#define UNR " + std::to_string(v.unr) + "\n" +
                                    "__kernel void q6x_ysum8(__global const float * y, __global float * ys, const int n8) {\n"
                                    "    const int m = get_global_id(0);\n"
                                    "    if (m < n8) { const float4 a = vload4(0, y + 8*m); const float4 b = vload4(0, y + 8*m + 4);\n"
                                    "        ys[m] = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)); }\n"
                                    "}\n" + q6r_source;
            if (rregs) {
                const char * src_ptr = src.c_str();
                cl_program p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
                CL_CHECK(err);
                const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -DQK_K=256 -cl-nv-verbose";
                CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
                size_t log_size = 0;
                CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
                std::string log(log_size, '\0');
                CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL));
                const size_t at = log.find("Compiling entry function 'q6r'");
                const size_t ur = at == std::string::npos ? std::string::npos : log.find("Used", at);
                printf("  %s: %s\n", name, ur == std::string::npos ? "?" : log.substr(ur, log.find('\n', ur) - ur).c_str());
                clReleaseProgram(p);
                continue;
            }
            cl_program p = build_program_from_source(context, device, src.c_str());
            cl_kernel ks = clCreateKernel(p, "q6x_ysum8", &err); CL_CHECK(err);
            cl_kernel k  = clCreateKernel(p, "q6r",       &err); CL_CHECK(err);
            CL_CHECK(clSetKernelArg(ks, 0, sizeof(cl_mem), &d_Y));
            CL_CHECK(clSetKernelArg(ks, 1, sizeof(cl_mem), &d_YS));
            CL_CHECK(clSetKernelArg(ks, 2, sizeof(cl_int), &n8_i));
            const size_t ys_local  = 64;
            const size_t ys_global = ((ncols / 8 + ys_local - 1) / ys_local) * ys_local;
            const double ys_ms = time_kernel(pq, ks, ys_global, ys_local, iters);
            CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &d_QR));
            CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &d_Y));
            CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &d_YS));
            CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &d_D));
            CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int), &ncols_i));
            CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_int), &nrows_i));
            CL_CHECK(clSetKernelArg(k, 6, sizeof(float) * v.r * v.lwg, NULL));
            CL_CHECK(clSetKernelArg(k, 7, sizeof(float) * yl_floats, NULL));
            CL_CHECK(clSetKernelArg(k, 8, sizeof(float) * ych / 8, NULL));
            const size_t rows_per_wg = v.lwg / 8 * v.r;
            const size_t global      = ((nrows + rows_per_wg - 1) / rows_per_wg) * v.lwg;
            std::fill(out.begin(), out.end(), 0.0f);
            CL_CHECK(clEnqueueWriteBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
            const double ms = time_kernel(pq, k, global, v.lwg, iters);
            CL_CHECK(clEnqueueReadBuffer(pq, d_D, true, 0, sizeof(float) * nrows, out.data(), 0, NULL, NULL));
            printf("  %s %6.3f ms %5.2f GB/s (err %.0e) | +ysum %.3f ms: %5.2f GB/s, x%.2f\n",
                   name, ms, q_bytes / (ms * 1e-3) / 1e9, max_rel_err(out, ref),
                   ys_ms, q_bytes / ((ms + ys_ms) * 1e-3) / 1e9, l8_ms / (ms + ys_ms));
            clReleaseKernel(k);
            clReleaseKernel(ks);
            clReleaseProgram(p);
        }
        clReleaseMemObject(d_QR);
    }
    if (!regs) {
        const double l8_ms2 = q6x_run_l8(pq, d_Q, d_Y, d_D, ncols, nrows, iters);
        printf("  integrated l8 again               %6.3f ms %5.2f GB/s\n", l8_ms2, q_bytes / (l8_ms2 * 1e-3) / 1e9);
    }

    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_YS);
    clReleaseMemObject(d_D);
}

int main(void) {
    // ggml_init fills the fp16 lookup tables used by the CPU reference (to_float)
    struct ggml_init_params params = { /* .mem_size = */ 1024*1024, /* .mem_buffer = */ NULL, /* .no_alloc = */ true };
    struct ggml_context * gctx = ggml_init(params);

    // the first OpenCL device of the registry (GGML_OPENCL_PLATFORM / GGML_OPENCL_DEVICE choose it)
    ggml_cl_probe_devices();
    if (g_cl_devices.empty() || !ggml_cl_activate(0)) {
        fprintf(stderr, "no OpenCL device\n");
        return 1;
    }

    // optionally dump the device binary (PTX on NVIDIA) of the backend program for inspection
    if (const char * dump_path = getenv("BENCH_DMMV_DUMP_PTX")) {
        size_t bin_size = 0;
        CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, NULL));
        std::vector<unsigned char> bin(bin_size);
        unsigned char * bin_ptr = bin.data();
        CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, NULL));
        FILE * f = fopen(dump_path, "wb");
        if (f) {
            fwrite(bin.data(), 1, bin.size(), f);
            fclose(f);
            printf("program binary (%zu bytes) written to %s\n", bin.size(), dump_path);
        }
        ggml_free(gctx);
        return 0;
    }

    cl_int err;
    cl_command_queue pq = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    bench_read_bandwidth(pq);

    // optionally print the ptxas register usage of the backend program and of the A/B reference
    if (getenv("BENCH_DMMV_REGS")) {
        const std::string opts = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math "
                                 "-DQK4_0=32 -DQR4_0=2 -DQK4_1=32 -DQR4_1=2 -DQK5_0=32 -DQR5_0=2 -DQK5_1=32 -DQR5_1=2 -DQK8_0=32 -DQR8_0=1 "
                                 "-DQK_K=256 -DK_QUANTS_PER_ITERATION=" + std::to_string(K_QUANTS_PER_ITERATION) + " -cl-nv-verbose";
        const std::string sources[2] = { generate_kernels(), ref_v4l_source };
        for (const std::string & src : sources) {
            const char * src_ptr = src.c_str();
            cl_program p = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
            CL_CHECK(err);
            CL_CHECK(clBuildProgram(p, 0, NULL, opts.c_str(), NULL, NULL));
            size_t log_size = 0;
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
            std::string log(log_size, '\0');
            CL_CHECK(clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL));
            printf("%s\n", log.c_str());
            clReleaseProgram(p);
        }
        ggml_free(gctx);
        return 0;
    }

    // BENCH_DMMV_Q6AB=1: interleaved A/B of the q6r kernel of the backend against q6rt
    if (getenv("BENCH_DMMV_Q6AB")) {
        ggml_cl_be_init();
        gpu_warmup(pq, 3.0);
        bench_q6_ab(pq, 2048, 248320, 3, 4, "(2B output)");
        bench_q6_ab(pq, 2048,    512, 40, 5, "(2B attn_v)");
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    // BENCH_DMMV_PATTERN=1: read-only access patterns (row-major against tiles of 32 rows)
    if (getenv("BENCH_DMMV_PATTERN")) {
        gpu_warmup(pq, 3.0);
        bench_patterns(pq);
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    // BENCH_DMMV_Q4AB=1: interleaved A/B of the Q4_K kernels, after warming up the GPU
    if (getenv("BENCH_DMMV_Q4AB")) {
        gpu_warmup(pq, 3.0);
        bench_q4_ab(pq, 2048, 6144, 10, 5, "(2B ffn_gate/ffn_up)");
        bench_q4_ab(pq, 2048, 2048, 20, 5, "(2B attn_gate/ssm_out)");
        bench_q4_ab(pq, 6144, 2048, 10, 5, "(2B ffn_down)");
        bench_q4_ab(pq, 2048,  512, 40, 5, "(2B attn_k)");
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    // BENCH_DMMV_Q45X=1: only the experimental Q4_K/Q5_K kernels (BENCH_DMMV_Q45X_REGS=1: their registers)
    if (getenv("BENCH_DMMV_Q45X")) {
        bench_q45x(pq, GGML_TYPE_Q4_K, 2048, 6144, 20, "(2B ffn_gate/ffn_up)");
        if (!getenv("BENCH_DMMV_Q45X_REGS")) {
            bench_q45x(pq, GGML_TYPE_Q4_K, 2048, 2048, 30, "(2B attn_gate/ssm_out)");
            bench_q45x(pq, GGML_TYPE_Q4_K, 6144, 2048, 20, "(2B ffn_down)");
            bench_q45x(pq, GGML_TYPE_Q4_K, 2048, 4096, 20, "(2B attn_q)");
            bench_q45x(pq, GGML_TYPE_Q4_K, 2048,  512, 50, "(2B attn_k)");
            bench_q45x(pq, GGML_TYPE_Q4_K, 2048,   16, 50, "(2B ssm_alpha/beta)");
        }
        bench_q45x(pq, GGML_TYPE_Q5_K, 2048, 6144, 20, "(2B attn_qkv)");
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    // BENCH_DMMV_Q6X=1: only the experimental Q6_K kernels (BENCH_DMMV_Q6X_REGS=1: their ptxas report)
    if (getenv("BENCH_DMMV_Q6X")) {
        if (getenv("BENCH_DMMV_Q6X_REGS")) {
            bench_q6x(pq, 2048, 512, 1, "(registers only)");
            clReleaseCommandQueue(pq);
            ggml_free(gctx);
            return 0;
        }
        bench_q6x(pq, 2048, 248320, 3, "(2B token_embd/output)");
        bench_q6x(pq, 6144,   2048, 20, "(2B ffn_down)");
        bench_q6x(pq, 2048,    512, 50, "(2B attn_v)");
        bench_q6x(pq, 9216,   2560, 20, "(4B ffn_down)");
        clReleaseCommandQueue(pq);
        ggml_free(gctx);
        return 0;
    }

    cl_program ref_prog = nullptr;
    if (getenv("BENCH_DMMV_AB")) {
        ref_prog = build_program_from_source(context, device, ref_v4l_source);
        ref_v4l_kernel = clCreateKernel(ref_prog, "ref_dmmv_q6_K_v4l", &err);
        CL_CHECK(err);
    }

    const bench_case cases[] = {
        { GGML_TYPE_Q4_K, 2048,   6144, 20, "(2B ffn_gate/ffn_up)"   },
        { GGML_TYPE_Q4_K, 6144,   2048, 20, "(2B ffn_down shape)"    },
        { GGML_TYPE_Q5_K, 2048,   6144, 20, "(2B ffn shape)"         },
        { GGML_TYPE_Q5_K, 6144,   2048, 20, "(2B ffn_down shape)"    },
        { GGML_TYPE_Q6_K, 2048,   6144, 20, "(2B ffn shape)"         },
        { GGML_TYPE_Q6_K, 6144,   2048, 20, "(2B ffn_down)"          },
        { GGML_TYPE_Q4_K, 2560,   9216, 20, "(4B ffn_gate/ffn_up)"   },
        { GGML_TYPE_Q6_K, 9216,   2560, 20, "(4B ffn_down)"          },
        { GGML_TYPE_Q6_K, 2048, 248320,  3, "(2B token_embd/output)" },
        { GGML_TYPE_Q4_K, 1536,   8960, 20, "(Qwen2.5-1.5B ffn)"     },
        { GGML_TYPE_Q6_K, 1536, 151936,  3, "(Qwen2.5-1.5B output)"  },
    };
    for (const auto & bc : cases) {
        bench_dmmv(pq, bc);
    }

    clReleaseCommandQueue(pq);
    ggml_free(gctx);
    return 0;
}
