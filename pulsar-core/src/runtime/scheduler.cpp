#include "pulsar/runtime/scheduler.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstring>

// Continuous-batching bookkeeping over the KV pool declared in pulsar/runtime/kv/active_buffer.hpp.

namespace pulsar {

Scheduler::Scheduler(ActiveBuffer& active, int64_t max_running_seqs, int64_t max_chunk_size)
    : active(active),
      max_running(max_running_seqs),
      max_chunk_size(max_chunk_size) {
    TORCH_CHECK(max_running_seqs > 0, "Scheduler: max_running_seqs must be > 0");
    TORCH_CHECK(max_chunk_size > 0, "Scheduler: max_chunk_size must be > 0");
}

void Scheduler::add_request(int64_t seq_id, int64_t prompt_len) {
    TORCH_CHECK(prompt_len > 0, "Scheduler: prompt_len must be > 0");
    TORCH_CHECK(this->seqs.find(seq_id) == this->seqs.end(), "Scheduler: seq ", seq_id, " already known");
    this->seqs[seq_id] = Seq{prompt_len, /*prefill_pos=*/0, /*prefilling=*/true};
    this->waiting.push_back(seq_id);
}

void Scheduler::grow_request(int64_t seq_id, int64_t new_prompt_len) {
    auto it = this->seqs.find(seq_id);
    TORCH_CHECK(it != this->seqs.end(), "Scheduler: grow_request unknown seq ", seq_id);
    Seq& s = it->second;
    TORCH_CHECK(
        s.prefilling && s.prefill_pos == 0,
        "Scheduler: grow_request seq ",
        seq_id,
        " has already started prefill"
    );
    TORCH_CHECK(
        new_prompt_len > s.prompt_len,
        "Scheduler: grow_request new_prompt_len ",
        new_prompt_len,
        " must exceed current ",
        s.prompt_len
    );
    s.prompt_len = new_prompt_len;
}

bool Scheduler::is_running(int64_t seq_id) const {
    return std::find(this->running.begin(), this->running.end(), seq_id) != this->running.end();
}

bool Scheduler::is_waiting(int64_t seq_id) const {
    return std::find(this->waiting.begin(), this->waiting.end(), seq_id) != this->waiting.end();
}

namespace {

// Pad a per-seq list of page-id vectors into an int32 [n, max_pages] table.
at::Tensor pad_page_tables(const std::vector<std::vector<int32_t>>& rows, at::Device dev) {
    const int64_t n = static_cast<int64_t>(rows.size());
    int64_t max_pages = 1;
    for (const auto& r : rows) {
        max_pages = std::max<int64_t>(max_pages, static_cast<int64_t>(r.size()));
    }
    auto t = at::zeros({n, max_pages}, at::TensorOptions().dtype(at::kInt));
    auto acc = t.accessor<int32_t, 2>();
    for (int64_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rows[i].size(); ++j) {
            acc[i][j] = rows[i][j];
        }
    }
    return t.to(dev);
}

}  // namespace

BatchDescriptor Scheduler::step() {
    BatchDescriptor desc;
    const auto dev = this->active.device();

    // Decode group: running sequences that have finished prefilling.
    std::vector<int64_t> dec_ids;
    std::vector<std::vector<int32_t>> dec_tables;
    std::vector<int32_t> dec_ctx;
    std::vector<int32_t> dec_slots;
    for (int64_t sid : this->running) {
        Seq& st = this->seqs.at(sid);
        if (st.prefilling) {
            continue;
        }
        const int64_t pos = this->active.active_len(sid);  // the new token lands here
        this->active.append(sid, 1);
        auto slot = this->active.slot_mapping(sid, pos, 1);
        dec_ids.push_back(sid);
        dec_tables.push_back(this->active.pages_of(sid));
        dec_ctx.push_back(static_cast<int32_t>(pos + 1));
        dec_slots.push_back(slot.item<int32_t>());
    }

    // Prefill group: continue in-progress prefills, then admit waiting.
    int64_t budget = this->max_chunk_size;
    std::vector<int64_t> pre_ids;
    std::vector<std::vector<int32_t>> pre_tables;
    std::vector<int32_t> pre_seqlens_k;
    std::vector<int32_t> pre_cu;  // prefix sums, starts at 0
    pre_cu.push_back(0);
    std::vector<int32_t> pre_slots;
    std::vector<int32_t> pre_prompt_starts;

    // Prompt-token index and RoPE/slot pos are DISTINCT: mid-prefill recall splices
    // grow active_len, so the slot/pos come from the buffer tail, not the prompt
    // index. With recall off they coincide (pos_start == prompt_start).
    auto do_prefill_chunk = [&](int64_t sid) {
        Seq& st = this->seqs.at(sid);
        const int64_t prompt_start = st.prefill_pos;  // prompt-token index into the prompt
        const int64_t remaining = st.prompt_len - prompt_start;
        const int64_t chunk = std::min(remaining, budget);
        if (chunk <= 0) {
            return false;
        }
        int64_t pos_start;
        if (!this->active.has_seq(sid)) {
            this->active.allocate(sid, chunk);
            pos_start = 0;
        } else {
            pos_start = this->active.active_len(sid);  // true tail (prior splices/evicts included)
            this->active.append(sid, chunk);
        }
        auto slot = this->active.slot_mapping(sid, pos_start, chunk);
        auto slot_cpu = slot.to(at::kCPU);
        auto sa = slot_cpu.accessor<int32_t, 1>();
        for (int64_t i = 0; i < chunk; ++i) {
            pre_slots.push_back(sa[i]);
        }
        pre_prompt_starts.push_back(static_cast<int32_t>(prompt_start));
        pre_ids.push_back(sid);
        pre_tables.push_back(this->active.pages_of(sid));
        pre_seqlens_k.push_back(static_cast<int32_t>(pos_start + chunk));
        pre_cu.push_back(pre_cu.back() + static_cast<int32_t>(chunk));
        st.prefill_pos += chunk;
        budget -= chunk;
        if (st.prefill_pos >= st.prompt_len) {
            st.prefilling = false;
        }
        return true;
    };

    // In-progress prefills already in running (FIFO by running order).
    for (int64_t sid : this->running) {
        if (!this->seqs.at(sid).prefilling || budget <= 0) {
            continue;
        }
        do_prefill_chunk(sid);
    }
    while (!this->waiting.empty() && this->num_running() < this->max_running && budget > 0) {
        const int64_t sid = this->waiting.front();
        const Seq& st = this->seqs.at(sid);
        const int64_t first_chunk = std::min(st.prompt_len - st.prefill_pos, budget);
        // Only admit if the first chunk's pages are available. A fresh seq
        // (prefill_pos == 0) allocates from scratch; a resumed seq (prefill_pos >
        // 0) already holds pages and only needs the append growth.
        const bool room = st.prefill_pos == 0
            ? this->active.can_allocate(first_chunk)
            : this->active.pages_to_append(sid, first_chunk) <= this->active.num_free_pages();
        if (!room) {
            break;
        }
        this->waiting.pop_front();
        this->running.push_back(sid);
        do_prefill_chunk(sid);
    }

    desc.prefill_seq_ids = c10::List<int64_t>(pre_ids);
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pre_cu.size())}, at::TensorOptions().dtype(at::kInt));
        if (!pre_cu.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), pre_cu.data(), pre_cu.size() * sizeof(int32_t));
        }
        desc.prefill_cu_seqlens_q = t.to(dev);
    }
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pre_seqlens_k.size())}, at::TensorOptions().dtype(at::kInt));
        if (!pre_seqlens_k.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), pre_seqlens_k.data(), pre_seqlens_k.size() * sizeof(int32_t));
        }
        desc.prefill_seqlens_k = t.to(dev);
    }
    desc.prefill_page_tables = pre_tables.empty() ? at::zeros({0, 1}, at::TensorOptions().dtype(at::kInt)).to(dev)
                                                  : pad_page_tables(pre_tables, dev);
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pre_slots.size())}, at::TensorOptions().dtype(at::kInt));
        if (!pre_slots.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), pre_slots.data(), pre_slots.size() * sizeof(int32_t));
        }
        desc.prefill_slot_mapping = t.to(dev);
    }
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pre_prompt_starts.size())}, at::TensorOptions().dtype(at::kInt));
        if (!pre_prompt_starts.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), pre_prompt_starts.data(), pre_prompt_starts.size() * sizeof(int32_t));
        }
        desc.prefill_prompt_starts = t.to(dev);
    }

    desc.decode_seq_ids = c10::List<int64_t>(dec_ids);
    {
        at::Tensor t = at::empty({static_cast<int64_t>(dec_ctx.size())}, at::TensorOptions().dtype(at::kInt));
        if (!dec_ctx.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), dec_ctx.data(), dec_ctx.size() * sizeof(int32_t));
        }
        desc.decode_context_lens = t.to(dev);
    }
    desc.decode_page_tables = dec_tables.empty() ? at::zeros({0, 1}, at::TensorOptions().dtype(at::kInt)).to(dev)
                                                 : pad_page_tables(dec_tables, dev);
    {
        at::Tensor t = at::empty({static_cast<int64_t>(dec_slots.size())}, at::TensorOptions().dtype(at::kInt));
        if (!dec_slots.empty()) {
            std::memcpy(t.data_ptr<int32_t>(), dec_slots.data(), dec_slots.size() * sizeof(int32_t));
        }
        desc.decode_slot_mapping = t.to(dev);
    }
    return desc;
}

void Scheduler::update(c10::List<int64_t> seq_ids, c10::List<int64_t> sampled_tokens, c10::List<bool> finished) {
    TORCH_CHECK(
        seq_ids.size() == sampled_tokens.size() && seq_ids.size() == finished.size(),
        "Scheduler.update: seq_ids/sampled_tokens/finished length mismatch"
    );
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        const int64_t sid = seq_ids.get(i);
        auto it = this->seqs.find(sid);
        if (it == this->seqs.end()) {
            continue;
        }
        if (finished.get(i)) {
            this->active.free(sid);
            this->running.erase(std::remove(this->running.begin(), this->running.end(), sid), this->running.end());
            this->seqs.erase(it);
        }
    }
}

void Scheduler::cancel(int64_t seq_id) {
    auto it = this->seqs.find(seq_id);
    if (it == this->seqs.end()) {
        return;
    }
    this->waiting.erase(std::remove(this->waiting.begin(), this->waiting.end(), seq_id), this->waiting.end());
    this->running.erase(std::remove(this->running.begin(), this->running.end(), seq_id), this->running.end());
    if (this->active.has_seq(seq_id)) {
        this->active.free(seq_id);
    }
    this->seqs.erase(it);
}

void Scheduler::park(int64_t seq_id) {
    auto it = this->seqs.find(seq_id);
    if (it == this->seqs.end()) {
        return;
    }
    this->waiting.erase(std::remove(this->waiting.begin(), this->waiting.end(), seq_id), this->waiting.end());
    this->running.erase(std::remove(this->running.begin(), this->running.end(), seq_id), this->running.end());
    this->seqs.erase(it);  // KV pages stay in the active buffer for a later resume.
}

void Scheduler::resume(int64_t seq_id, int64_t resume_addr, int64_t prompt_len) {
    TORCH_CHECK(this->active.has_seq(seq_id), "Scheduler: resume needs active KV for seq ", seq_id);
    TORCH_CHECK(this->seqs.find(seq_id) == this->seqs.end(), "Scheduler: seq ", seq_id, " already known");
    TORCH_CHECK(
        resume_addr >= 0 && prompt_len > resume_addr,
        "Scheduler: resume prompt_len ",
        prompt_len,
        " must exceed resume_addr ",
        resume_addr
    );
    // prefill_pos is the conversation ADDRESS the re-prefill sources token ids from;
    // the RoPE/pool position comes separately from the window tail
    // (active.active_len) in do_prefill_chunk.
    this->seqs[seq_id] = Seq{
        prompt_len,
        /*prefill_pos=*/resume_addr,
        /*prefilling=*/true
    };
    this->waiting.push_back(seq_id);
}

}  // namespace pulsar
