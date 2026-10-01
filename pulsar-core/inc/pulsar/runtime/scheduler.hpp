#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"

#include <ATen/core/Tensor.h>

#include <ATen/core/List.h>

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

// Continuous-batching scheduler over the ActiveBuffer (pulsar/runtime/kv/active_buffer.hpp), and the
// batched descriptors it produces for the paged attention ops (write_kv, attn_prefill,
// attn_decode). Does NOT run a model forward: the caller owns Q/K/V compute and
// sampling. Engine-internal plain C++, not TorchBind-exposed; unit tests in
// pulsar-core/tests/runtime/.

namespace pulsar {

// One step's batched work: a prefill group and a decode group, each mapping onto
// one attention op. All tensors live on the pool's device.
//
// Caller contract per step (mass_decay is the caller's own fp32 [num_seqs] EMA rate
// per sequence, in *_seq_ids order; the scheduler holds no session knobs):
//   Prefill: assemble q for the prefill query tokens in prefill_seq_ids order
//     (prefill_cu_seqlens_q tokens per seq), compute k/v, then
//       write_kv(k_pool, v_pool, k, v, prefill_slot_mapping)
//       attn_prefill(q, k_pool, v_pool, mass_pool,
//           prefill_page_tables, prefill_cu_seqlens_q, prefill_seqlens_k, scale,
//           mass_decay)
//   Decode: assemble q (one token per decode_seq_ids entry), compute k/v, then
//       write_kv(k_pool, v_pool, k, v, decode_slot_mapping)
//       attn_decode(q, k_pool, v_pool, mass_pool,
//           decode_page_tables, decode_context_lens, scale, mass_decay)
struct BatchDescriptor {
    c10::List<int64_t> prefill_seq_ids;
    at::Tensor prefill_cu_seqlens_q;  // int32 [num_prefill + 1]
    at::Tensor prefill_seqlens_k;  // int32 [num_prefill]
    at::Tensor prefill_page_tables;  // int32 [num_prefill, max_pages]
    at::Tensor prefill_slot_mapping;  // int32 [total_prefill_q]
    at::Tensor prefill_prompt_starts;  // int32 [num_prefill], conversation addr each chunk begins at

    c10::List<int64_t> decode_seq_ids;
    at::Tensor decode_context_lens;  // int32 [num_decode]
    at::Tensor decode_page_tables;  // int32 [num_decode, max_pages]
    at::Tensor decode_slot_mapping;  // int32 [num_decode]

    int64_t num_prefill_tokens() const {
        return this->prefill_slot_mapping.numel();
    }
    int64_t num_decode_tokens() const {
        return this->decode_slot_mapping.numel();
    }
};

// Continuous-batching scheduler over an ActiveBuffer. Bookkeeping only: it
// admits waiting prefills (chunked), continues running decodes, retires finished
// sequences, and produces a BatchDescriptor per step. It does NOT run the model.
// Eviction is the Engine's concern (density evict -> recall); the scheduler never
// evicts.
struct Scheduler {
    // Holds a reference to the active buffer (the Engine owns it). max_chunk_size caps
    // the prefill query tokens admitted per step (chunked prefill splits long
    // prompts).
    Scheduler(ActiveBuffer& active, int64_t max_running_seqs, int64_t max_chunk_size);

    void add_request(int64_t seq_id, int64_t prompt_len);
    // Extend a waiting request's prompt_len before its prefill starts.
    void grow_request(int64_t seq_id, int64_t new_prompt_len);
    BatchDescriptor step();
    // seq_ids: sequences that produced a token this step (decodes + prompts just
    // completed). sampled_tokens: their next input tokens; only the length is read,
    // the Engine owns the ids. finished: retire + free pages for the matching sequence.
    void update(c10::List<int64_t> seq_ids, c10::List<int64_t> sampled_tokens, c10::List<bool> finished);

    // Drop a sequence immediately: remove it from waiting/running, free its KV
    // pages, and forget it. No-op on an unknown sequence.
    void cancel(int64_t seq_id);

    // Multi-turn: retire a sequence from scheduling but KEEP its KV in the active
    // buffer (cancel minus active.free). Remove it from waiting/running and forget its
    // Seq state, leaving its pages for a later resume. No-op on an unknown sequence.
    void park(int64_t seq_id);

    // Multi-turn: re-admit a parked sequence (KV still active) as a waiting prefill.
    // resume_addr is the conversation ADDRESS the new turn re-prefills from; the
    // prefill covers addresses [resume_addr, prompt_len), attending over the active KV.
    // resume_addr is distinct from active.active_len (a window count): the RoPE/pool
    // position comes from the window tail, the token identity from the address. Requires
    // active KV and the sequence not already known; prompt_len must exceed resume_addr.
    void resume(int64_t seq_id, int64_t resume_addr, int64_t prompt_len);

    int64_t num_waiting() const {
        return static_cast<int64_t>(this->waiting.size());
    }
    int64_t num_running() const {
        return static_cast<int64_t>(this->running.size());
    }
    bool is_running(int64_t seq_id) const;
    bool is_waiting(int64_t seq_id) const;

  private:
    struct Seq {
        int64_t prompt_len;
        int64_t prefill_pos;  // prompt tokens prefilled so far
        bool prefilling;  // still has prompt tokens to prefill
    };

    ActiveBuffer& active;
    int64_t max_running;
    int64_t max_chunk_size;

    std::deque<int64_t> waiting;
    std::vector<int64_t> running;
    std::unordered_map<int64_t, Seq> seqs;
};

}  // namespace pulsar
