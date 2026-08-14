#include "pulsar/model/compiled_model.hpp"

#include "pulsar/runtime/kv/active_buffer.hpp"  // ActiveBuffer (paged KV pools)

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace pulsar {

namespace {

// n_layers = 1 + max i over the package's constant FQNs matching "layers.<i>.".
int64_t derive_n_layers(torch::inductor::AOTIModelPackageLoader& loader) {
    const std::regex re("layers\\.([0-9]+)\\.");
    int64_t max_layer = -1;
    for (const std::string& fqn : loader.get_constant_fqns()) {
        std::smatch m;
        if (std::regex_search(fqn, m, re)) {
            max_layer = std::max<int64_t>(max_layer, std::stoll(m[1].str()));
        }
    }
    TORCH_CHECK(max_layer >= 0, "CompiledModel: no layer constants found in the .pt2 package");
    return max_layer + 1;
}

void* current_stream() {
    return reinterpret_cast<void*>(c10::cuda::getCurrentCUDAStream().stream());
}

at::Tensor i32_to_dev(const std::vector<int32_t>& v, at::Device dev) {
    auto t = at::empty({static_cast<int64_t>(v.size())}, at::TensorOptions().dtype(at::kInt));
    if (!v.empty()) {
        std::memcpy(t.data_ptr<int32_t>(), v.data(), v.size() * sizeof(int32_t));
    }
    return t.to(dev);
}

at::Tensor i64_to_dev(const std::vector<int64_t>& v, at::Device dev) {
    auto t = at::empty({static_cast<int64_t>(v.size())}, at::TensorOptions().dtype(at::kLong));
    if (!v.empty()) {
        std::memcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t));
    }
    return t.to(dev);
}

// Row index [0, 1, ..., R-1, 0, 0, ..., 0] of length num_seqs on `dev`: the real
// rows in order, then the padding rows all mapped to row 0. index_select with it
// row-pads a per-seq tensor up to num_seqs by duplicating row 0.
at::Tensor pad_row_index(int64_t R, int64_t num_seqs, at::Device dev) {
    std::vector<int64_t> idx(num_seqs, 0);
    for (int64_t i = 0; i < R; ++i) {
        idx[i] = i;
    }
    return i64_to_dev(idx, dev);
}

// Right-pad a [rows, held] page table's column dim to max_pages with page id 0.
// held <= max_pages (a sequence never holds more than the baked width). Padded
// columns lie past ceil(active_len/page_size) and are never read by the attn kernels.
at::Tensor pad_page_table_cols(at::Tensor bt, int64_t max_pages) {
    int64_t held = bt.size(1);
    TORCH_CHECK(
        held <= max_pages,
        "page_tables has ",
        held,
        " columns but the compiled graph bakes max_pages=",
        max_pages
    );
    if (held < max_pages) {
        bt = at::constant_pad_nd(bt, {0, max_pages - held}, 0);
    }
    return bt.contiguous();
}

// Physical pool slots row 0 attends: slot(j) = bt_row0[j / page_size] * page_size
// + j % page_size, for j in [0, len). len is row 0's context (decode) or its key
// length seqlens_k[0] (prefill). Same physical set in every layer.
std::vector<int64_t> row0_attended_slots(const at::Tensor& bt_row0, int64_t len, int64_t page_size) {
    auto bt_cpu = bt_row0.to(at::kCPU, at::kInt).contiguous();
    auto a = bt_cpu.accessor<int32_t, 1>();
    std::vector<int64_t> slots(len);
    for (int64_t j = 0; j < len; ++j) {
        slots[j] = static_cast<int64_t>(a[j / page_size]) * page_size + (j % page_size);
    }
    return slots;
}

}  // namespace

CompiledModel::CompiledModel(
    std::string decode_pt2,
    std::string prefill_pt2,
    int64_t num_seqs,
    int64_t num_pages,
    int64_t page_size
)
    : decode_loader(decode_pt2),
      prefill_loader(prefill_pt2),
      num_seqs(num_seqs),
      num_pages(num_pages),
      page_size(page_size),
      n_layers(derive_n_layers(decode_loader)),
      max_pages(0) {
    TORCH_CHECK(num_seqs > 0, "CompiledModel: num_seqs must be > 0");
    // Paged export lays out num_pages = max_pages * num_seqs, so the baked
    // page-table width is num_pages / num_seqs.
    TORCH_CHECK(
        num_pages % num_seqs == 0,
        "CompiledModel: num_pages (",
        num_pages,
        ") must be divisible by num_seqs (",
        num_seqs,
        ")"
    );
    this->max_pages = num_pages / num_seqs;
}

void CompiledModel::validate(ActiveBuffer& kv) const {
    if (kv.num_pages() != this->num_pages || kv.page_size() != this->page_size || kv.n_layers() != this->n_layers) {
        TORCH_CHECK(
            false,
            "allocator dims do not match the compiled graph: got "
            "(n_layers=",
            kv.n_layers(),
            ", num_pages=",
            kv.num_pages(),
            ", page_size=",
            kv.page_size(),
            "), baked (n_layers=",
            this->n_layers,
            ", num_pages=",
            this->num_pages,
            ", page_size=",
            this->page_size,
            ")"
        );
    }
}

at::Tensor CompiledModel::run_decode_group(
    torch::inductor::AOTIModelPackageLoader& loader,
    const GroupBatch& g,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& mass
) {
    TORCH_CHECK(g.context_lens.defined(), "decode group missing context_lens");
    const int64_t R = g.context_lens.numel();
    TORCH_CHECK(
        R >= 1 && R <= this->num_seqs,
        "decode batch has ",
        R,
        " sequences; the compiled graph bakes num_seqs=",
        this->num_seqs,
        " (need 1 <= R <= num_seqs)"
    );
    const at::Device dev = g.tokens.device();
    const int64_t pad = this->num_seqs - R;

    at::Tensor tokens = g.tokens, pos = g.pos;
    at::Tensor slot = g.slot_mapping, context_lens = g.context_lens;
    at::Tensor page_tables = g.page_tables;

    // Snapshot row 0's attended-slot mass before the run; the (1 + pad)x over-count
    // from the duplicated rows is divided back down afterwards.
    std::vector<int64_t> slots0;
    at::Tensor slots0_dev;
    std::vector<at::Tensor> pre_mass;
    if (pad > 0) {
        const int64_t ctx0 = g.context_lens[0].item<int32_t>();
        slots0 = row0_attended_slots(g.page_tables[0], ctx0, this->page_size);
        slots0_dev = i64_to_dev(slots0, dev);
        pre_mass.resize(this->n_layers);
        for (int64_t l = 0; l < this->n_layers; ++l) {
            pre_mass[l] = mass.select(0, l)
                              .view({this->num_pages * this->page_size, mass.size(3)})
                              .index_select(0, slots0_dev)
                              .clone();
        }

        at::Tensor idx = pad_row_index(R, this->num_seqs, dev);
        tokens = tokens.index_select(0, idx);
        pos = pos.index_select(0, idx);
        slot = slot.index_select(0, idx);
        context_lens = context_lens.index_select(0, idx);
        page_tables = page_tables.index_select(0, idx);
    }
    page_tables = pad_page_table_cols(page_tables, this->max_pages);

    // decode: [tokens, pos, slot_mapping, page_tables, context_lens, k, v, mass]
    std::vector<at::Tensor> inputs{tokens, pos, slot, page_tables, context_lens, k, v, mass};
    at::Tensor logits = loader.run(inputs, current_stream())[0];

    if (pad > 0) {
        const double div = static_cast<double>(1 + pad);
        for (int64_t l = 0; l < this->n_layers; ++l) {
            auto flat = mass.select(0, l).view({this->num_pages * this->page_size, mass.size(3)});
            auto post = flat.index_select(0, slots0_dev);
            auto corrected = pre_mass[l] + (post - pre_mass[l]) / div;
            flat.index_copy_(0, slots0_dev, corrected);
        }
    }
    return logits.narrow(0, 0, R);  // real rows only
}

at::Tensor CompiledModel::run_prefill_group(
    torch::inductor::AOTIModelPackageLoader& loader,
    const GroupBatch& g,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& mass
) {
    TORCH_CHECK(g.cu_seqlens_q.defined() && g.seqlens_k.defined(), "prefill group missing cu_seqlens_q/seqlens_k");
    const int64_t R = g.cu_seqlens_q.numel() - 1;
    TORCH_CHECK(
        R >= 1 && R <= this->num_seqs,
        "prefill batch has ",
        R,
        " sequences; the compiled graph bakes num_seqs=",
        this->num_seqs,
        " (need 1 <= R <= num_seqs)"
    );
    const at::Device dev = g.tokens.device();
    const int64_t total_q = g.tokens.numel();
    const int64_t pad = this->num_seqs - R;

    at::Tensor tokens = g.tokens, pos = g.pos;
    at::Tensor slot = g.slot_mapping;
    at::Tensor cu = g.cu_seqlens_q, seqlens_k = g.seqlens_k;
    at::Tensor page_tables = g.page_tables;

    std::vector<int64_t> slots0;
    at::Tensor slots0_dev;
    std::vector<at::Tensor> pre_mass;
    if (pad > 0) {
        auto cu_cpu = cu.to(at::kCPU, at::kInt).contiguous();
        auto cu_a = cu_cpu.accessor<int32_t, 1>();
        const int64_t total_q0 = cu_a[1] - cu_a[0];  // seq 0's query token count
        const int64_t sk0 = g.seqlens_k[0].item<int32_t>();  // seq 0's key length

        // Snapshot seq 0's attended key slots' mass (over-counted by the dummies).
        slots0 = row0_attended_slots(g.page_tables[0], sk0, this->page_size);
        slots0_dev = i64_to_dev(slots0, dev);
        pre_mass.resize(this->n_layers);
        for (int64_t l = 0; l < this->n_layers; ++l) {
            pre_mass[l] = mass.select(0, l)
                              .view({this->num_pages * this->page_size, mass.size(3)})
                              .index_select(0, slots0_dev)
                              .clone();
        }

        // Query-token index: the real total_q rows, then `pad` copies of seq 0's
        // query rows [0, total_q0). Dummies appended after the real rows keep the
        // real logits in [0, total_q).
        std::vector<int64_t> q_idx(total_q);
        for (int64_t i = 0; i < total_q; ++i) {
            q_idx[i] = i;
        }
        for (int64_t d = 0; d < pad; ++d) {
            for (int64_t j = 0; j < total_q0; ++j) {
                q_idx.push_back(j);
            }
        }
        at::Tensor q_idx_dev = i64_to_dev(q_idx, dev);
        tokens = tokens.index_select(0, q_idx_dev);
        pos = pos.index_select(0, q_idx_dev);
        slot = slot.index_select(0, q_idx_dev);

        // cu_seqlens_q: extend the prefix sums by total_q0 per dummy -> [num_seqs+1].
        std::vector<int32_t> cu_v(cu_a.data(), cu_a.data() + R + 1);
        int32_t last = cu_a[R];
        for (int64_t d = 0; d < pad; ++d) {
            last += static_cast<int32_t>(total_q0);
            cu_v.push_back(last);
        }
        cu = i32_to_dev(cu_v, dev);

        at::Tensor ridx = pad_row_index(R, this->num_seqs, dev);
        seqlens_k = seqlens_k.index_select(0, ridx);
        page_tables = page_tables.index_select(0, ridx);
    }
    page_tables = pad_page_table_cols(page_tables, this->max_pages);

    // prefill: [tokens, pos, slot_mapping, page_tables, cu_seqlens_q, seqlens_k, k, v, mass]
    std::vector<at::Tensor> inputs{tokens, pos, slot, page_tables, cu, seqlens_k, k, v, mass};
    at::Tensor logits = loader.run(inputs, current_stream())[0];

    if (pad > 0) {
        const double div = static_cast<double>(1 + pad);
        for (int64_t l = 0; l < this->n_layers; ++l) {
            auto flat = mass.select(0, l).view({this->num_pages * this->page_size, mass.size(3)});
            auto post = flat.index_select(0, slots0_dev);
            auto corrected = pre_mass[l] + (post - pre_mass[l]) / div;
            flat.index_copy_(0, slots0_dev, corrected);
        }
    }
    return logits.narrow(0, 0, total_q);  // real query rows only
}

StepLogits CompiledModel::forward(const StepBatch& batch, ActiveBuffer& kv) {
    this->validate(kv);
    // Fetch the stacked pools once; the graphs mutate them in place.
    at::Tensor k = kv.k_pool_all();
    at::Tensor v = kv.v_pool_all();
    at::Tensor mass = kv.mass_pool_all();
    StepLogits out;
    if (batch.prefill.has_value()) {
        out.prefill = this->run_prefill_group(this->prefill_loader, *batch.prefill, k, v, mass);
    }
    if (batch.decode.has_value()) {
        out.decode = this->run_decode_group(this->decode_loader, *batch.decode, k, v, mass);
    }
    return out;
}

}  // namespace pulsar
