#include "pulsar/ops.hpp"

#include <torch/library.h>

// The schema strings are the ABI shared by C++ and Python (pulsar/ops.py loads
// this library and calls torch.ops.pulsar.*).

TORCH_LIBRARY(pulsar, m) {
    m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def(
        "attn_causal(Tensor q, Tensor k, Tensor v, "
        "Tensor attention_mass, float scale, int causal_offset) -> (Tensor, Tensor)"
    );
    m.def(
        "attn_causal_cache(Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor(a!) attention_mass, float scale, int cur_len) -> Tensor"
    );
    m.def(
        "write_kv(Tensor(a!) k_pool, Tensor(b!) v_pool, Tensor k_new, "
        "Tensor v_new, Tensor slot_mapping) -> ()"
    );
    m.def(
        "attn_decode(Tensor q, Tensor k_pool, Tensor v_pool, "
        "Tensor(a!) mass_pool, Tensor page_tables, Tensor context_lens, "
        "float scale, Tensor attention_mass_decay, "
        "Tensor(b!)? lse_capture=None, float mass_length_gain=0.) -> Tensor"
    );
    m.def(
        "attn_decode_scalar(Tensor q, Tensor k_pool, "
        "Tensor v_pool, Tensor(a!) mass_pool, Tensor page_tables, "
        "Tensor context_lens, float scale, "
        "Tensor attention_mass_decay, float mass_length_gain=0.) -> Tensor"
    );
    m.def(
        "attn_decode_split(Tensor q, Tensor k_pool, "
        "Tensor v_pool, Tensor(a!) mass_pool, Tensor page_tables, "
        "Tensor context_lens, float scale, int num_splits, "
        "Tensor attention_mass_decay, "
        "Tensor(b!)? lse_capture=None, float mass_length_gain=0.) -> Tensor"
    );
    m.def(
        "attn_prefill(Tensor q, Tensor k_pool, Tensor v_pool, "
        "Tensor(a!) mass_pool, Tensor page_tables, Tensor cu_seqlens_q, "
        "Tensor seqlens_k, float scale, Tensor attention_mass_decay, "
        "Tensor(b!)? lse_capture=None, float mass_length_gain=0.) -> Tensor"
    );
    m.def(
        "attn_prefill_scalar(Tensor q, Tensor k_pool, "
        "Tensor v_pool, Tensor(a!) mass_pool, Tensor page_tables, "
        "Tensor cu_seqlens_q, Tensor seqlens_k, float scale, "
        "Tensor attention_mass_decay, float mass_length_gain=0.) -> Tensor"
    );
    m.def(
        "gemm_w4a16(Tensor x, Tensor weight_packed, Tensor weight_scale, "
        "int group_size) -> Tensor"
    );
    m.def(
        "gemm_w4a16_split(Tensor x, Tensor weight_packed, Tensor weight_scale, "
        "int group_size, int num_splits) -> Tensor"
    );
    m.def(
        "repack_w4a16(Tensor weight_packed, Tensor weight_scale, int group_size) "
        "-> (Tensor, Tensor)"
    );
    m.def(
        "gemm_w4a16_marlin(Tensor x, Tensor weight_marlin, Tensor scale_marlin, "
        "int group_size) -> Tensor"
    );
    m.def("rope(Tensor x, Tensor pos, float theta) -> Tensor");
    m.def(
        "reposition_kv(Tensor(a!) k_pool, Tensor slots, Tensor old_positions, "
        "Tensor new_positions, float theta) -> ()"
    );
}

TORCH_LIBRARY_IMPL(pulsar, CUDA, m) {
    m.impl("rmsnorm", TORCH_FN(pulsar::rmsnorm_cuda));
    m.impl("attn_causal", TORCH_FN(pulsar::attn_causal_cuda));
    m.impl("attn_causal_cache", TORCH_FN(pulsar::attn_causal_cache_cuda));
    m.impl("write_kv", TORCH_FN(pulsar::write_kv_cuda));
    m.impl("attn_decode", TORCH_FN(pulsar::attn_decode_cuda));
    m.impl("attn_decode_scalar", TORCH_FN(pulsar::attn_decode_scalar_cuda));
    m.impl("attn_decode_split", TORCH_FN(pulsar::attn_decode_split_cuda));
    m.impl("attn_prefill", TORCH_FN(pulsar::attn_prefill_cuda));
    m.impl("attn_prefill_scalar", TORCH_FN(pulsar::attn_prefill_scalar_cuda));
    m.impl("gemm_w4a16", TORCH_FN(pulsar::gemm_w4a16_cuda));
    m.impl("gemm_w4a16_split", TORCH_FN(pulsar::gemm_w4a16_split_cuda));
    m.impl("repack_w4a16", TORCH_FN(pulsar::repack_w4a16_cuda));
    m.impl("gemm_w4a16_marlin", TORCH_FN(pulsar::gemm_w4a16_marlin_cuda));
    m.impl("rope", TORCH_FN(pulsar::rope_cuda));
    m.impl("reposition_kv", TORCH_FN(pulsar::reposition_kv_cuda));
}
