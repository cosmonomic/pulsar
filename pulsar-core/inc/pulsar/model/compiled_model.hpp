#pragma once

#include "pulsar/model/interface.hpp"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cstdint>
#include <string>

// AOTInductor-backed `Model` running the compiled decode/prefill .pt2 graphs over
// the paged KV pools. The graphs bake num_seqs, pool capacity, and head/layer dims
// static; validate() checks the runtime allocator matches. The stacked pools are
// graph inputs mutated in place.
//
// Static-batch padding: the .pt2 bakes num_seqs and page_tables=[num_seqs,
// max_pages] static (only prefill total_q is dynamic), so a group with R < num_seqs
// rows is padded up to num_seqs by duplicating row 0. The dummy rows' logits are
// discarded. Row 0's attention-mass side-output accumulates (atomicAdd), so each
// dummy over-counts row 0's attended slots by one; forward snapshots those slots
// and divides the delta back down to match an unpadded run. Padding stays inside
// this class (the native Qwen2Model does not pad).
//
// Engine-internal plain C++, not TorchBind-exposed.

namespace pulsar {

struct ActiveBuffer;  // defined in active_buffer.hpp

struct CompiledModel : Model {
    // decode_pt2 / prefill_pt2: the compiled .pt2 package paths. num_seqs /
    // num_pages / page_size are the graphs' baked dims; n_layers is derived from
    // the decode package's baked layer constants.
    CompiledModel(
        std::string decode_pt2,
        std::string prefill_pt2,
        int64_t num_seqs,
        int64_t num_pages,
        int64_t page_size
    );

    StepLogits forward(const StepBatch& batch, ActiveBuffer& kv) override;

  private:
    // Guard the runtime allocator against the graph's baked pool shapes.
    void validate(ActiveBuffer& kv) const;
    // Pad the group to the baked num_seqs / [num_seqs, max_pages] page table,
    // run the graph, and return logits for the real rows only (decode: the first
    // R rows; prefill: the first total_q query rows). When R < num_seqs the mass
    // over-count from the duplicated row is corrected in place.
    at::Tensor run_decode_group(
        torch::inductor::AOTIModelPackageLoader& loader,
        const GroupBatch& g,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& mass
    );
    at::Tensor run_prefill_group(
        torch::inductor::AOTIModelPackageLoader& loader,
        const GroupBatch& g,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& mass
    );

    torch::inductor::AOTIModelPackageLoader decode_loader;
    torch::inductor::AOTIModelPackageLoader prefill_loader;
    int64_t num_seqs;
    int64_t num_pages;
    int64_t page_size;
    int64_t n_layers;
    int64_t max_pages;  // baked page-table width = num_pages / num_seqs
};

}  // namespace pulsar
