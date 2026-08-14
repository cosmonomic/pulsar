#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"  // DemotedKV

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <vector>

// Host-side page helpers shared by the runtime: the whole-page decode of a DemotedKV
// that every spill bucket reserves from, and the coldest-first ranking every spill
// bucket drains to budget by. The buckets stay independent implementations of the
// same concept; these are free functions they share, not a base class.
//
// Internal to pulsar-core/src/runtime; not part of the installed headers.

namespace pulsar {

// One demoted page decoded out of a DemotedKV: its permanent identity, its token
// ids, the eviction-time mass snapshot, and the ActiveBuffer slot block its K/V is
// copied out of.
struct DemotedPage {
    int64_t page_id = 0;
    std::vector<int64_t> token_ids;
    double mass = 0.0;
    int64_t src_base = 0;  // active-pool slot block base
};

// Decode a DemotedKV into whole pages, host side only. Empty when nothing was
// demoted. `who` prefixes the check messages, naming the calling bucket's reserve.
//
// Eviction demotes full pages, so two invariants hold per page and are checked, not
// assumed: the page's addresses are one aligned block [base, base + page_size), so
// page index = base / page_size; and its victim slots are a contiguous
// page_size-run in the active pool (demote emits page * page_size + offset), so the
// page's source base is its first slot.
std::vector<DemotedPage> demoted_pages(const DemotedKV& demoted, int64_t page_size, const char* who);

// A held page's rank input: the frozen mass snapshot and the step it was taken at.
struct PageMass {
    double mass = 0.0;
    int64_t mass_step = 0;
};

// Indices into `pages`, coldest first, by lazy-decayed effective mass
// eff = mass * (1-decay)^(step - mass_step) with the exponent clamped >= 0. The sort
// is stable, so pages passed in page-id order get an ascending page-id tie-break.
std::vector<int64_t> coldest_first(const std::vector<PageMass>& pages, int64_t step, double decay);

}  // namespace pulsar
