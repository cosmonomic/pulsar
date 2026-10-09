#pragma once

#include <c10/macros/Macros.h>

#include <cuda_runtime.h>

#include <cstdint>

// The per-batch inputs every paged attention kernel reads, and the per-key arithmetic
// they share: a key's RoPE position in its sequence's view, its rotation, and its slot
// in the mass output.
//
// The pool holds UNROTATED keys. A key's position is a function of its index j in the
// sequence's view (page table order) alone, so a line shared by several views takes
// each view's own position, and the kernels rotate every key as they read it. Queries
// arrive already rotated.

namespace pulsar {
namespace attn {

template <typename scalar_t> struct PagedAttnParams {
    const scalar_t* q;  // [total_q, n_q_heads, head_dim], rotated
    const scalar_t* k_pool;  // [num_lines, page_size, n_kv_heads, head_dim], unrotated
    const scalar_t* v_pool;  // [num_lines, page_size, n_kv_heads, head_dim]
    const int32_t* page_tables;  // [num_seqs, max_pages]
    const int32_t* cu_seqlens_q;  // [num_seqs + 1]; null when each sequence has one query
    const int32_t* seqlens_k;  // [num_seqs]
    const int32_t* rope_layout;  // [num_seqs, 3], see KeyLayout
    const uint64_t* rope_turns;  // [head_dim / 2], see rope_turns_per_position
    float* mass;  // [total_view_pages, page_size, n_q_heads]; null skips the mass pass
    const int32_t* cu_view_pages;  // [num_seqs + 1]
    const float* mass_decay;  // [num_seqs]
    float mass_length_gain;
    scalar_t* o;  // [total_q, n_q_heads, head_dim]
    float* lse;  // [total_q, n_q_heads]; null when not captured
    int num_seqs;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int max_pages;
    int group;
    int total_view_pages;
    float scale;
};

// The RoPE position of key index j in one sequence's view:
//   j < n_sink                 -> j
//   n_sink <= j < working_lo   -> short_offset
//   j >= working_lo            -> short_offset + 1 + j - working_lo
// {0, 0, -1} is the identity.
struct KeyLayout {
    int n_sink;
    int working_lo;
    int short_offset;

    __device__ __forceinline__ int position(int j) const {
        if (j < this->n_sink) {
            return j;
        }
        if (j < this->working_lo) {
            return this->short_offset;
        }
        return this->short_offset + 1 + j - this->working_lo;
    }
};

__device__ __forceinline__ KeyLayout key_layout(const int32_t* rope_layout, int s) {
    const int32_t* row = rope_layout + 3 * static_cast<int64_t>(s);
    return KeyLayout{row[0], row[1], row[2]};
}

// cos/sin of the RoPE angle position * turns_per_position, rounded to fp32.
//
// The product wraps mod 2^64, which drops the whole turns exactly and leaves the
// angle's fraction of a turn to 2^-64. Its top 32 bits split into the nearest quarter
// turn and a remainder within an eighth of a turn of it, and only that remainder goes
// through the hardware __sincosf, whose absolute error on it is under 2^-21.
__device__ __forceinline__ void
rope_sincos_turns(int position, uint64_t turns_per_position, float& cos_out, float& sin_out) {
    constexpr float kRadiansPerUnit = 1.46291807926715968e-9f;  // 2 pi / 2^32
    const uint64_t turns = static_cast<uint64_t>(static_cast<int64_t>(position)) * turns_per_position;
    const uint32_t top = static_cast<uint32_t>(turns >> 32);
    const uint32_t quadrant = (top + (1u << 29)) >> 30;
    const int32_t remainder = static_cast<int32_t>(top - (quadrant << 30));

    float sine, cosine;
    __sincosf(static_cast<float>(remainder) * kRadiansPerUnit, &sine, &cosine);
    switch (quadrant & 3) {
    case 0:
        cos_out = cosine;
        sin_out = sine;
        break;
    case 1:
        cos_out = -sine;
        sin_out = cosine;
        break;
    case 2:
        cos_out = -cosine;
        sin_out = -sine;
        break;
    default:
        cos_out = sine;
        sin_out = -cosine;
        break;
    }
}

template <typename scalar_t> struct RotatedPair {
    scalar_t lo;
    scalar_t hi;
};

// The rotate_half rotation of one frequency pair (x[i], x[i + head_dim/2]), rounded to
// the pool dtype. The rounding modes are explicit so that every kernel holding the
// same key computes the same bits.
template <typename scalar_t>
__device__ __forceinline__ RotatedPair<scalar_t> rope_rotate(float lo, float hi, float cos_a, float sin_a) {
    return RotatedPair<scalar_t>{
        static_cast<scalar_t>(__fsub_rn(__fmul_rn(lo, cos_a), __fmul_rn(hi, sin_a))),
        static_cast<scalar_t>(__fadd_rn(__fmul_rn(hi, cos_a), __fmul_rn(lo, sin_a))),
    };
}

// Sequence s's rows of the mass output: key index j's row is cu_view_pages[s] *
// page_size + j, so two views sharing a line accumulate apart.
template <typename scalar_t>
__device__ __forceinline__ float* seq_mass(const PagedAttnParams<scalar_t>& p, int s, int ctx_len, int page_size) {
    const int first = p.cu_view_pages[s];
    const int end = p.cu_view_pages[s + 1];
    CUDA_KERNEL_ASSERT(first >= 0 && (end - first) * page_size >= ctx_len && end <= p.total_view_pages);
    return p.mass + static_cast<int64_t>(first) * page_size * p.n_q_heads;
}

}  // namespace attn
}  // namespace pulsar
