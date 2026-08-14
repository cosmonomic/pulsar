#include "pulsar/runtime/kv/active_buffer.hpp"

#include "pulsar/ops.hpp"
#include "../tensor_util.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>

// The KV pool declared in pulsar/runtime/kv/active_buffer.hpp. Page allocation, identity, the density
// eviction and the recall recompaction; the paged kernels own the reads.

namespace pulsar {
namespace {

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

double ema_bias_correction(double decay, int64_t age) {
    if (decay <= 0.0 || age <= 0) {
        return 1.0;
    }
    return std::max(1.0 - std::pow(1.0 - decay, static_cast<double>(age)), 1e-6);
}

}  // namespace

ActiveBuffer::ActiveBuffer(
    int64_t n_layers,
    int64_t num_pages,
    int64_t page_size,
    int64_t n_kv_heads,
    int64_t n_q_heads,
    int64_t head_dim,
    at::ScalarType dtype,
    std::string device,
    double rope_theta
)
    : num_layers(n_layers),
      page_count(num_pages),
      page_len(page_size),
      n_kv_heads(n_kv_heads),
      n_q_heads(n_q_heads),
      head_dim(head_dim),
      rope_theta(rope_theta),
      dev(at::Device(device)) {
    TORCH_CHECK(
        n_layers > 0 && num_pages > 0 && page_size > 0 && n_kv_heads > 0 && head_dim > 0,
        "ActiveBuffer: dims must be positive"
    );
    TORCH_CHECK(
        n_q_heads >= n_kv_heads && n_q_heads % n_kv_heads == 0,
        "ActiveBuffer: n_q_heads (",
        n_q_heads,
        ") must be a positive multiple of n_kv_heads (",
        n_kv_heads,
        ")"
    );
    TORCH_CHECK(dtype != at::ScalarType::Undefined, "ActiveBuffer: dtype must be set (float16/bfloat16/float32)");
    auto opts = at::TensorOptions().dtype(dtype).device(this->dev);
    this->k_pools = at::zeros({n_layers, num_pages, page_size, n_kv_heads, head_dim}, opts);
    this->v_pools = at::zeros({n_layers, num_pages, page_size, n_kv_heads, head_dim}, opts);
    this->mass_pools = at::zeros(
        {n_layers, num_pages, page_size, n_q_heads},
        at::TensorOptions().dtype(at::kFloat).device(this->dev)
    );
    this->free_list.reserve(num_pages);
    // Hand pages out low-first: push high ids first so pop_back yields 0,1,2...
    for (int64_t b = num_pages - 1; b >= 0; --b) {
        this->free_list.push_back(static_cast<int32_t>(b));
    }
    // Relevance layers default to ALL layers until set_relevance_layers overrides.
    std::vector<int64_t> all_layers(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        all_layers[l] = l;
    }
    this->set_relevance_layers(std::move(all_layers));
}

int64_t ActiveBuffer::rope_of(const RopeLayout& layout, int64_t index) const {
    // Contiguous: the slot index IS the position. Also the state a sequence is in before
    // its first cycle, when there is no distant region to collapse yet.
    if (layout.position_layout != PositionLayout::compacted || layout.working_lo < 0) {
        return index;
    }
    if (index < layout.n_sink) {
        return index;
    }
    if (index < layout.working_lo) {
        return layout.short_offset;
    }
    return layout.short_offset + 1 + (index - layout.working_lo);
}

ActiveBuffer::RopeLayout ActiveBuffer::layout_for(
    int64_t ctx,
    int64_t n_sink,
    int64_t n_working,
    PositionLayout layout,
    int64_t short_offset
) const {
    RopeLayout out;
    out.position_layout = layout;
    if (layout != PositionLayout::compacted) {
        return out;  // working_lo stays -1
    }
    out.n_sink = std::min(n_sink, ctx);
    out.working_lo = ActiveBuffer::working_start(ctx, n_sink, n_working);
    out.short_offset = short_offset;
    return out;
}

ActiveBuffer::RopeLayout ActiveBuffer::layout_of(int64_t seq_id) const {
    auto it = this->rope_layout.find(seq_id);
    return it == this->rope_layout.end() ? RopeLayout{} : it->second;
}

int64_t ActiveBuffer::rope_pos_at(int64_t seq_id, int64_t index) const {
    return this->rope_of(this->layout_of(seq_id), index);
}

void ActiveBuffer::set_relevance_layers(std::vector<int64_t> layers) {
    if (layers.empty()) {
        this->relevance_offset = 0;
        this->relevance_count = 0;
        return;
    }
    for (size_t i = 1; i < layers.size(); ++i) {
        TORCH_CHECK(
            layers[i] == layers[i - 1] + 1,
            "set_relevance_layers: layers must be a contiguous ascending "
            "run, got ",
            at::IntArrayRef(layers)
        );
    }
    TORCH_CHECK(
        layers.front() >= 0 && layers.back() < this->num_layers,
        "set_relevance_layers: layers ",
        at::IntArrayRef(layers),
        " out of range [0, ",
        this->num_layers,
        ")"
    );
    this->relevance_offset = layers.front();
    this->relevance_count = static_cast<int64_t>(layers.size());
}

std::vector<int32_t> ActiveBuffer::claim_pages(int64_t n) {
    TORCH_CHECK(
        static_cast<int64_t>(this->free_list.size()) >= n,
        "ActiveBuffer: out of pages (need ",
        n,
        ", free ",
        this->free_list.size(),
        ")"
    );
    std::vector<int32_t> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t b = this->free_list.back();
        this->free_list.pop_back();
        // A reused page may hold stale mass; zero it in EVERY layer.
        this->mass_pools.select(1, b).zero_();
        out.push_back(b);
    }
    return out;
}

bool ActiveBuffer::can_allocate(int64_t num_tokens) const {
    return static_cast<int64_t>(this->free_list.size()) >= ceil_div(num_tokens, this->page_len);
}

void ActiveBuffer::allocate(int64_t seq_id, int64_t num_tokens) {
    TORCH_CHECK(this->tables.find(seq_id) == this->tables.end(), "ActiveBuffer: seq ", seq_id, " already allocated");
    TORCH_CHECK(num_tokens >= 0, "ActiveBuffer: num_tokens must be >= 0");
    const int64_t npages = ceil_div(num_tokens, this->page_len);
    this->tables[seq_id] = this->claim_pages(npages);
    this->lens[seq_id] = num_tokens;
    // Identity arrays track the page table 1:1; -1/empty until set_identity fills.
    this->page_ids[seq_id] = std::vector<int64_t>(npages, -1);
    this->tok[seq_id] = std::vector<std::vector<int64_t>>(npages);
    this->next_pidx[seq_id] = 0;
}

int64_t ActiveBuffer::pages_to_append(int64_t seq_id, int64_t num_new_tokens) const {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    const int64_t new_len = this->lens.at(seq_id) + num_new_tokens;
    const int64_t need = ceil_div(new_len, this->page_len);
    const int64_t held = static_cast<int64_t>(it->second.size());
    return std::max<int64_t>(0, need - held);
}

void ActiveBuffer::append(int64_t seq_id, int64_t num_new_tokens) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(num_new_tokens >= 0, "ActiveBuffer: num_new_tokens must be >= 0");
    const int64_t new_len = this->lens[seq_id] + num_new_tokens;
    const int64_t need = ceil_div(new_len, this->page_len);
    const int64_t held = static_cast<int64_t>(it->second.size());
    if (need > held) {
        auto more = this->claim_pages(need - held);
        it->second.insert(it->second.end(), more.begin(), more.end());
        this->page_ids.at(seq_id).resize(need, -1);
        this->tok.at(seq_id).resize(need);
    }
    this->lens[seq_id] = new_len;
}

void ActiveBuffer::free(int64_t seq_id) {
    auto it = this->tables.find(seq_id);
    if (it == this->tables.end()) {
        return;
    }
    for (int32_t b : it->second) {
        this->free_list.push_back(b);
    }
    this->tables.erase(it);
    this->lens.erase(seq_id);
    this->page_ids.erase(seq_id);
    this->tok.erase(seq_id);
    this->next_pidx.erase(seq_id);
    this->rope_layout.erase(seq_id);
}

bool ActiveBuffer::has_seq(int64_t seq_id) const {
    return this->tables.find(seq_id) != this->tables.end();
}

const std::vector<int32_t>& ActiveBuffer::pages_of(int64_t seq_id) const {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    return it->second;
}

at::Tensor ActiveBuffer::page_table(int64_t seq_id) const {
    const std::vector<int32_t>& pages = this->pages_of(seq_id);
    at::Tensor t = at::empty({static_cast<int64_t>(pages.size())}, at::TensorOptions().dtype(at::kInt));
    if (!pages.empty()) {
        std::memcpy(t.data_ptr<int32_t>(), pages.data(), pages.size() * sizeof(int32_t));
    }
    return t.to(this->dev);
}

int64_t ActiveBuffer::active_len(int64_t seq_id) const {
    auto it = this->lens.find(seq_id);
    TORCH_CHECK(it != this->lens.end(), "ActiveBuffer: unknown seq ", seq_id);
    return it->second;
}

at::Tensor ActiveBuffer::slot_mapping(int64_t seq_id, int64_t start_pos, int64_t num_tokens) const {
    const auto& pages = this->pages_of(seq_id);
    TORCH_CHECK(start_pos >= 0 && num_tokens >= 0, "ActiveBuffer: slot_mapping needs non-negative range");
    TORCH_CHECK(
        start_pos + num_tokens <= this->lens.at(seq_id),
        "ActiveBuffer: slot_mapping range [",
        start_pos,
        ", ",
        start_pos + num_tokens,
        ") exceeds active_len ",
        this->lens.at(seq_id)
    );
    std::vector<int32_t> slots;
    slots.reserve(num_tokens);
    for (int64_t p = start_pos; p < start_pos + num_tokens; ++p) {
        const int32_t page = pages[p / this->page_len];
        const int32_t off = static_cast<int32_t>(p % this->page_len);
        slots.push_back(page * static_cast<int32_t>(this->page_len) + off);
    }
    at::Tensor t = at::empty({static_cast<int64_t>(slots.size())}, at::TensorOptions().dtype(at::kInt));
    if (!slots.empty()) {
        std::memcpy(t.data_ptr<int32_t>(), slots.data(), slots.size() * sizeof(int32_t));
    }
    return t.to(this->dev);
}

DemotedKV ActiveBuffer::evict(
    int64_t seq_id,
    int64_t n_sink,
    int64_t n_working,
    PositionLayout layout,
    int64_t short_offset,
    double min_density,
    int64_t active_cap,
    int64_t block_radius,
    int64_t stream_len,
    double attention_mass_decay,
    std::vector<float>* cand_density,
    DensityByPosition* position_profile,
    EvictAttribution* attribution
) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        this->relevance_count > 0,
        "ActiveBuffer::evict: no relevance layers, so no mass was "
        "accumulated to rank pages by"
    );
    TORCH_CHECK(n_sink >= 0 && n_working >= 0, "ActiveBuffer: n_sink and n_working must be >= 0");
    const int64_t ctx = this->lens[seq_id];
    const auto& pages = it->second;
    const int64_t P = static_cast<int64_t>(pages.size());
    const int64_t page_len = this->page_len;
    const int64_t band_lo = std::min(n_sink, ctx);
    const int64_t band_hi = std::max<int64_t>(0, ctx - n_working);

    std::vector<char> is_victim(P, 0);
    std::unordered_map<int64_t, float> mass_of;

    // EVICTABLE pages: FULL pages entirely inside the band [band_lo, band_hi) --
    // outside the sink prefix and the recent working window.
    std::vector<int64_t> candidates;
    for (int64_t j = 0; j < P; ++j) {
        const int64_t p_lo = j * page_len;
        const int64_t p_hi = std::min((j + 1) * page_len, ctx);
        if (p_hi - p_lo == page_len && p_lo >= band_lo && p_hi <= band_hi) {
            candidates.push_back(j);
        }
    }
    if (candidates.empty()) {
        return this->demote_victim_pages(seq_id, is_victim, mass_of, n_sink, n_working, layout, short_offset);
    }

    // Per-page ranking mass over ALL logical pages in ADDRESS order (page j covers
    // [j*page_len, ...)): MEAN over the query heads FIRST, then MAX over the page's
    // slots, then MAX over the RELEVANCE layers. The head axis must collapse before the
    // slot max so the page's score belongs to a SINGLE token; the two do not commute.
    // Mass is stored per query head with no group mean, so each entry is already an
    // attention share as a multiple of uniform over the length the attention kernels
    // stated it against, and so is every reduction of it.
    // The relevance layers must be contiguous (set_relevance_layers enforces it) for
    // narrow to stay a view rather than copying the pool.
    auto relevant = this->mass_pools.narrow(0, this->relevance_offset, this->relevance_count);
    auto per_slot = relevant.mean(at::IntArrayRef{-1});  // [layers, num_pages, page_len]
    auto per_layer = per_slot.amax({-1});  // [layers, num_pages]
    auto page_mass = per_layer.amax({0});  // [num_pages] fp32
    std::vector<int64_t> all_phys(P);
    for (int64_t j = 0; j < P; ++j) {
        all_phys[j] = pages[j];
    }
    at::Tensor all_phys_t = at::empty({static_cast<int64_t>(all_phys.size())}, at::TensorOptions().dtype(at::kLong));
    if (!all_phys.empty()) {
        std::memcpy(all_phys_t.data_ptr<int64_t>(), all_phys.data(), all_phys.size() * sizeof(int64_t));
    }
    auto raw_mass = page_mass.index_select(0, all_phys_t.to(this->dev));  // [P]

    // Divide out the EMA's startup bias, for age measured from the page's ABSOLUTE
    // address. The eps floor means the quotient is NOT bounded by 1, so neither is the
    // bar it feeds. The undivided reduction is the share of a FULL active buffer the
    // page holds, not of the context it was earned over, so a page reads the same
    // whenever it earned it.
    const double decay = attention_mass_decay;
    const auto& pids = this->page_ids.at(seq_id);
    std::vector<float> denom(P);
    std::vector<int64_t> page_age(P);
    for (int64_t j = 0; j < P; ++j) {
        const int64_t age = stream_len - pids[j] * page_len;
        page_age[j] = age;
        denom[j] = static_cast<float>(ema_bias_correction(decay, age));
    }
    at::Tensor corrected = raw_mass / at::tensor(denom, at::TensorOptions().dtype(at::kFloat)).to(this->dev);
    // A configured reference length is restored HERE, not in the attention kernels: the
    // mass is a sum over query tokens and a constant commutes with that sum, so one
    // multiply per page puts the density in its units. The products round differently
    // from a per-query multiply inside the kernel, so this is exact-arithmetic equal to
    // that and not bit-equal to it.
    if (this->mass_reference_length > 0) {
        corrected = corrected * this->density_length_gain();
    }

    // The two base rows come back first; the neighbourhood is built on the host because
    // its window is a PAGE ID range, which a fixed-width pool over positions cannot
    // express once the page table is gappy.
    auto dens = at::stack({raw_mass, corrected}).to(at::kCPU, at::kFloat).contiguous();
    auto rows = dens.accessor<float, 2>();

    // NEIGHBOURHOOD MAX in PAGE ID distance: a page takes the highest value among pages
    // whose id is within block_radius of its own. There is no block structure -- every
    // page carries the max of its own neighbourhood -- so a page survives the bar exactly
    // when some page within the radius is above it.
    //
    // The window is page id, not page POSITION: ids ascend strictly but not contiguously
    // (check_page_aligned), so after an eviction two adjacent positions can be thousands
    // of tokens apart in the conversation. KVMemory::select_recall_pages bounds by page
    // id for the same reason.
    //
    // ids ascend, so each window is a contiguous position range whose ends both advance
    // monotonically with j -- a two-pointer sweep with a decreasing-value deque, O(P).
    auto neighbourhood_max = [&](const float* row) {
        std::vector<float> out(static_cast<size_t>(P));
        if (block_radius <= 0 || P <= 1) {
            std::copy(row, row + P, out.begin());
            return out;
        }
        std::deque<int64_t> best;  // positions, row values decreasing
        int64_t lo = 0, hi = 0;
        for (int64_t j = 0; j < P; ++j) {
            for (; hi < P && pids[hi] <= pids[j] + block_radius; ++hi) {
                while (!best.empty() && row[best.back()] <= row[hi]) {
                    best.pop_back();
                }
                best.push_back(hi);
            }
            for (; lo < P && pids[lo] < pids[j] - block_radius; ++lo) {
                if (!best.empty() && best.front() == lo) {
                    best.pop_front();
                }
            }
            // j is always inside its own window, so best is never empty here.
            out[static_cast<size_t>(j)] = row[best.front()];
        }
        return out;
    };
    const std::vector<float> neighbourhood_density = neighbourhood_max(&rows[1][0]);
    for (int64_t j : candidates) {
        mass_of[j] = neighbourhood_density[j];
    }

    if (position_profile) {
        for (int64_t j = 0; j < P; ++j) {
            const int64_t bucket = DensityByPosition::bucket_of(j, P);
            position_profile->raw_mass_sum[bucket] += rows[0][j];
            position_profile->corrected_density_sum[bucket] += rows[1][j];
            ++position_profile->page_count[bucket];
        }
    }

    // Report each candidate's density (candidate order) for the caller's histogram.
    if (cand_density) {
        cand_density->clear();
        cand_density->reserve(candidates.size());
        for (int64_t j : candidates) {
            cand_density->push_back(neighbourhood_density[j]);
        }
    }

    // Keep-bar: demote each candidate whose NEIGHBOURHOOD density is below min_density.
    // The bar is absolute, so a shorter or longer active length self-equilibrates around
    // it.
    int64_t bar_victims = 0;
    for (int64_t j : candidates) {
        if (static_cast<double>(neighbourhood_density[j]) < min_density) {
            is_victim[j] = 1;
            ++bar_victims;
        }
    }

    // Backstop: if the active length would still exceed active_cap, demote additional
    // LOWEST-density evictable pages toward active_cap. Bounded by the evictable band,
    // so the cap is a target, not a guarantee.
    //
    // Ranking is per page on the neighbourhood density, so a whole neighbourhood shares
    // one value and ties are the common case, not the exception. Ties break by AGE,
    // oldest first. Page POSITION is the tie-break rather than page_id: the two agree
    // (page indices ascend strictly with position, see check_page_aligned) and position
    // is defined even for a caller that never recorded identity.
    int64_t backstop_victims = 0;
    if (active_cap > 0) {
        const int64_t new_len = ctx - bar_victims * page_len;
        if (new_len > active_cap) {
            const int64_t need = ceil_div(new_len - active_cap, page_len);
            std::vector<int64_t> rem;
            for (int64_t j : candidates) {
                if (!is_victim[j]) {
                    rem.push_back(j);
                }
            }
            std::sort(rem.begin(), rem.end(), [&](int64_t a, int64_t b) {
                if (mass_of[a] != mass_of[b]) {
                    return mass_of[a] < mass_of[b];
                }
                return a < b;
            });
            backstop_victims = std::min(need, static_cast<int64_t>(rem.size()));
            for (int64_t i = 0; i < backstop_victims; ++i) {
                is_victim[rem[i]] = 1;
            }
        }
    }

    if (attribution) {
        *attribution = EvictAttribution{};
        attribution->bar_pages = bar_victims;
        attribution->backstop_pages = backstop_victims;
        attribution->candidate_age.reserve(candidates.size());
        for (int64_t j : candidates) {
            attribution->candidate_age.push_back(page_age[j]);
            if (is_victim[j]) {
                attribution->victim_age.push_back(page_age[j]);
            }
        }
        // Counterfactual: the same number of victims, ranked by the UNCORRECTED
        // neighbourhood mass, ties broken by position as the backstop breaks them.
        const std::vector<float> neighbourhood_raw_mass = neighbourhood_max(&rows[0][0]);
        const int64_t taken = bar_victims + backstop_victims;
        std::vector<int64_t> by_raw_mass = candidates;
        std::sort(by_raw_mass.begin(), by_raw_mass.end(), [&](int64_t a, int64_t b) {
            if (neighbourhood_raw_mass[a] != neighbourhood_raw_mass[b]) {
                return neighbourhood_raw_mass[a] < neighbourhood_raw_mass[b];
            }
            return a < b;
        });
        attribution->raw_mass_victims = taken;
        for (int64_t i = 0; i < taken; ++i) {
            if (is_victim[by_raw_mass[i]]) {
                ++attribution->raw_mass_agree;
            }
        }
    }
    return this->demote_victim_pages(seq_id, is_victim, mass_of, n_sink, n_working, layout, short_offset);
}

void ActiveBuffer::relayout(
    int64_t seq_id,
    int64_t n_sink,
    int64_t n_working,
    PositionLayout layout,
    int64_t short_offset,
    const std::vector<int64_t>& old_index
) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        short_offset > 0,
        "relayout: short_offset (",
        short_offset,
        ") must be a resolved absolute position > 0"
    );
    const int64_t ctx = this->lens[seq_id];
    TORCH_CHECK(
        old_index.empty() || static_cast<int64_t>(old_index.size()) == ctx,
        "relayout: old_index has ",
        old_index.size(),
        " entries for an active length of ",
        ctx
    );
    const auto& pages = it->second;
    const int64_t page_len = this->page_len;
    const RopeLayout old_layout = this->layout_of(seq_id);
    const RopeLayout new_layout = this->layout_for(ctx, n_sink, n_working, layout, short_offset);

    // Re-rope on the POSITION moving, not the slot index: a token can hold its slot and
    // still change position (the working run slid out from under it), and it can move
    // slots without changing position (both ends are the distant region's one value).
    std::vector<int32_t> moved_slots, from_positions, to_positions;
    for (int64_t index = 0; index < ctx; ++index) {
        const int64_t was = old_index.empty() ? index : old_index[index];
        const int64_t from = this->rope_of(old_layout, was);
        const int64_t to = this->rope_of(new_layout, index);
        if (from == to) {
            continue;
        }
        moved_slots.push_back(static_cast<int32_t>(pages[index / page_len] * page_len + index % page_len));
        from_positions.push_back(static_cast<int32_t>(from));
        to_positions.push_back(static_cast<int32_t>(to));
    }
    if (!moved_slots.empty()) {
        at::Tensor slots_dev = at::empty(
            {static_cast<int64_t>(moved_slots.size())},
            at::TensorOptions().dtype(at::kInt)
        );
        if (!moved_slots.empty()) {
            std::memcpy(slots_dev.data_ptr<int32_t>(), moved_slots.data(), moved_slots.size() * sizeof(int32_t));
        }
        slots_dev = slots_dev.to(this->dev);
        at::Tensor from_dev = at::empty(
            {static_cast<int64_t>(from_positions.size())},
            at::TensorOptions().dtype(at::kInt)
        );
        if (!from_positions.empty()) {
            std::memcpy(from_dev.data_ptr<int32_t>(), from_positions.data(), from_positions.size() * sizeof(int32_t));
        }
        from_dev = from_dev.to(this->dev);
        at::Tensor to_dev = at::empty({static_cast<int64_t>(to_positions.size())}, at::TensorOptions().dtype(at::kInt));
        if (!to_positions.empty()) {
            std::memcpy(to_dev.data_ptr<int32_t>(), to_positions.data(), to_positions.size() * sizeof(int32_t));
        }
        to_dev = to_dev.to(this->dev);
        for (int64_t l = 0; l < this->num_layers; ++l) {
            reposition_kv_cuda(this->k_pools.select(0, l), slots_dev, from_dev, to_dev, this->rope_theta);
        }
    }
    this->rope_layout[seq_id] = new_layout;
}

DemotedKV ActiveBuffer::demote_victim_pages(
    int64_t seq_id,
    const std::vector<char>& is_victim,
    const std::unordered_map<int64_t, float>& mass_of,
    int64_t n_sink,
    int64_t n_working,
    PositionLayout layout,
    int64_t short_offset
) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        short_offset > 0,
        "demote_victim_pages: short_offset (",
        short_offset,
        ") must be a resolved absolute position > 0"
    );
    const int64_t ctx = this->lens[seq_id];
    const auto& pages = it->second;
    const int64_t P = static_cast<int64_t>(pages.size());
    const int64_t page_len = this->page_len;
    TORCH_CHECK(
        static_cast<int64_t>(is_victim.size()) == P,
        "demote_victim_pages: victim mask out of sync with page table"
    );

    // Empty payload, returned as-is when nothing is evicted.
    DemotedKV demoted;
    demoted.addr_data = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    demoted.tok_data = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    demoted.mass_data = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
    demoted.victim_slots = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(this->dev));
    bool any_victim = false;
    for (int64_t j = 0; j < P; ++j) {
        if (is_victim[j]) {
            any_victim = true;
            break;
        }
    }
    if (!any_victim) {
        return demoted;  // nothing to evict
    }

    // One pass over logical pos [0, ctx): collect victim slots/pos for the demote
    // payload, and for each survivor the OLD slot index it holds now. Survivor slot
    // indices close up to [0, new_len) in order.
    std::vector<int64_t> victim_slots;
    std::vector<int32_t> victim_pos_i32;
    std::vector<int64_t> victim_addr, victim_tok;  // identity of the evicted tokens
    std::vector<float> victim_page_mass;  // one mass per evicted page (j order)
    std::vector<int64_t> survivor_old_index;
    std::vector<int32_t> survivor_pages_in_order;
    const auto& vic_pidx = this->page_ids.at(seq_id);
    const auto& vic_tok = this->tok.at(seq_id);
    int64_t victim_pages_count = 0;
    for (int64_t j = 0; j < P; ++j) {
        if (is_victim[j]) {
            ++victim_pages_count;
        }
    }
    const int64_t new_len = ctx - victim_pages_count * page_len;
    const RopeLayout old_layout = this->layout_of(seq_id);
    for (int64_t j = 0; j < P; ++j) {
        const int64_t p_lo = j * page_len;
        const int64_t p_hi = std::min((j + 1) * page_len, ctx);
        if (is_victim[j]) {
            auto mit = mass_of.find(j);
            victim_page_mass.push_back(mit != mass_of.end() ? mit->second : 0.0f);
            for (int64_t pos = p_lo; pos < p_hi; ++pos) {
                victim_slots.push_back(pages[j] * page_len + (pos - p_lo));
                // The BAKED position the rotation to short_offset starts from. Under a
                // compacted layout that is not the slot index.
                victim_pos_i32.push_back(static_cast<int32_t>(this->rope_of(old_layout, pos)));
                // Address/token id (whole-page-aligned): page_id*ps + offset.
                // Guard callers that never recorded identity (page_id == -1 / empty tok).
                const int64_t off = pos - p_lo;
                victim_addr.push_back(vic_pidx[j] >= 0 ? vic_pidx[j] * page_len + off : -1);
                victim_tok.push_back(off < static_cast<int64_t>(vic_tok[j].size()) ? vic_tok[j][off] : -1);
            }
        } else {
            survivor_pages_in_order.push_back(pages[j]);
            for (int64_t pos = p_lo; pos < p_hi; ++pos) {
                survivor_old_index.push_back(pos);
            }
        }
    }
    TORCH_CHECK(
        static_cast<int64_t>(survivor_old_index.size()) == new_len,
        "demote_victim_pages: survivor count ",
        survivor_old_index.size(),
        " != the whole-page surviving length ",
        new_len
    );

    // Rotate each victim's K in place from its baked position to short_offset, before
    // the gather. V is pos-invariant and untouched. The victim slots are freed below,
    // so mutating them here is harmless.
    if (!victim_slots.empty()) {
        std::vector<int32_t> victim_slots_i32(victim_slots.begin(), victim_slots.end());
        std::vector<int32_t> victim_short(victim_pos_i32.size(), static_cast<int32_t>(short_offset));
        at::Tensor vslots_dev = at::empty(
            {static_cast<int64_t>(victim_slots_i32.size())},
            at::TensorOptions().dtype(at::kInt)
        );
        if (!victim_slots_i32.empty()) {
            std::memcpy(
                vslots_dev.data_ptr<int32_t>(),
                victim_slots_i32.data(),
                victim_slots_i32.size() * sizeof(int32_t)
            );
        }
        vslots_dev = vslots_dev.to(this->dev);
        at::Tensor vold_dev = at::empty(
            {static_cast<int64_t>(victim_pos_i32.size())},
            at::TensorOptions().dtype(at::kInt)
        );
        if (!victim_pos_i32.empty()) {
            std::memcpy(vold_dev.data_ptr<int32_t>(), victim_pos_i32.data(), victim_pos_i32.size() * sizeof(int32_t));
        }
        vold_dev = vold_dev.to(this->dev);
        at::Tensor vshort_dev = at::empty(
            {static_cast<int64_t>(victim_short.size())},
            at::TensorOptions().dtype(at::kInt)
        );
        if (!victim_short.empty()) {
            std::memcpy(vshort_dev.data_ptr<int32_t>(), victim_short.data(), victim_short.size() * sizeof(int32_t));
        }
        vshort_dev = vshort_dev.to(this->dev);
        for (int64_t l = 0; l < this->num_layers; ++l) {
            reposition_kv_cuda(this->k_pools.select(0, l), vslots_dev, vold_dev, vshort_dev, this->rope_theta);
        }
    }

    // Record the victim slot references (into the flat pool) and identity; the single
    // copy out is the receiving bucket's fill, reading these slots before they are
    // reclaimed. The rotation above left every slot at short_offset.
    {
        at::Tensor t = at::empty({static_cast<int64_t>(victim_addr.size())}, at::TensorOptions().dtype(at::kLong));
        if (!victim_addr.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), victim_addr.data(), victim_addr.size() * sizeof(int64_t));
        }
        demoted.addr_data = t.to(at::Device(at::kCPU));
    }
    {
        at::Tensor t = at::empty({static_cast<int64_t>(victim_tok.size())}, at::TensorOptions().dtype(at::kLong));
        if (!victim_tok.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), victim_tok.data(), victim_tok.size() * sizeof(int64_t));
        }
        demoted.tok_data = t.to(at::Device(at::kCPU));
    }
    demoted.mass_data = at::tensor(victim_page_mass, at::TensorOptions().dtype(at::kFloat));
    {
        at::Tensor t = at::empty({static_cast<int64_t>(victim_slots.size())}, at::TensorOptions().dtype(at::kLong));
        if (!victim_slots.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), victim_slots.data(), victim_slots.size() * sizeof(int64_t));
        }
        demoted.victim_slots = t.to(this->dev);
    }

    // Free the victim pages and install the survivor table (original order).
    for (int64_t j = 0; j < P; ++j) {
        if (is_victim[j]) {
            this->free_list.push_back(pages[j]);
        }
    }
    it->second = std::move(survivor_pages_in_order);
    this->lens[seq_id] = new_len;
    // The table has closed up, so survivor positions are read at the old slot indices.
    this->relayout(seq_id, n_sink, n_working, layout, short_offset, survivor_old_index);

    // Rebuild identity to the survivor pages in order: page indices are
    // IMMUTABLE, a survivor keeps its page_id/tok verbatim. Whole-page eviction =>
    // a page-position filter.
    auto& seq_pidx = this->page_ids.at(seq_id);
    auto& seq_tok = this->tok.at(seq_id);
    TORCH_CHECK(
        static_cast<int64_t>(seq_pidx.size()) == P && static_cast<int64_t>(seq_tok.size()) == P,
        "demote_victim_pages: identity arrays out of sync with page table"
    );
    std::vector<int64_t> keep_pidx;
    std::vector<std::vector<int64_t>> keep_tok;
    for (int64_t j = 0; j < P; ++j) {
        if (is_victim[j]) {
            continue;
        }
        keep_pidx.push_back(seq_pidx[j]);
        keep_tok.push_back(std::move(seq_tok[j]));
    }
    seq_pidx = std::move(keep_pidx);
    seq_tok = std::move(keep_tok);
    return demoted;
}

void ActiveBuffer::recompact_with(
    int64_t seq_id,
    int64_t n_sink,
    int64_t n_working,
    PositionLayout layout,
    int64_t short_offset,
    const RecompactPlan& plan,
    const std::vector<at::Tensor>& recalled_k,
    const std::vector<at::Tensor>& recalled_v,
    int64_t stream_len,
    double attention_mass_decay
) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        short_offset > 0,
        "recompact_with: short_offset (",
        short_offset,
        ") must be a resolved absolute position > 0"
    );
    const int64_t N = static_cast<int64_t>(plan.survivor.size());
    TORCH_CHECK(
        static_cast<int64_t>(plan.old_pos.size()) == N && static_cast<int64_t>(plan.old_rope.size()) == N &&
            static_cast<int64_t>(plan.recalled_row.size()) == N && static_cast<int64_t>(plan.addr.size()) == N &&
            static_cast<int64_t>(plan.tok.size()) == N && static_cast<int64_t>(plan.score.size()) == N,
        "recompact_with: plan vectors must share length N"
    );
    TORCH_CHECK(
        static_cast<int64_t>(recalled_k.size()) == this->num_layers &&
            static_cast<int64_t>(recalled_v.size()) == this->num_layers,
        "recompact_with: recalled K/V must have one tensor per layer"
    );
    const int64_t page_len = this->page_len;
    const std::vector<int32_t> old_pages = it->second;  // survivor pages (permuted)

    // Pages are ATOMIC: a survivor page keeps its tokens in their physical slots and
    // only its slot index changes; a recalled page is written into a freshly claimed
    // page. No survivor K/V is copied. Plan is page-aligned: each block of page_len new
    // slot indices is one whole survivor page (offsets preserved) or one recalled page.
    // Only the last page may be partial (the survivor tail).
    const int64_t n_new_pages = ceil_div(N, page_len);
    std::vector<int32_t> new_pages(n_new_pages);
    std::vector<int32_t> recalled_dst_slots;  // fresh slots to write recalled K/V
    std::vector<int64_t> recalled_rows;  // matching rows into recalled_k/v
    std::vector<float> recalled_seed;  // per recalled slot: its page's seed s
    std::vector<int32_t> all_dst_slots(N), old_rope_i32(N), new_pos_i32(N);
    const double decay = attention_mass_decay;
    const RopeLayout merged_layout = this->layout_for(N, n_sink, n_working, layout, short_offset);

    for (int64_t np = 0; np < n_new_pages; ++np) {
        const int64_t lo = np * page_len;
        const int64_t hi = std::min(lo + page_len, N);
        if (plan.survivor[lo]) {
            // Whole survivor page reused in place; only its slot index shifts (the
            // re-rope pass below applies the position that follows from it).
            const int64_t old_page_pos = plan.old_pos[lo] / page_len;
            TORCH_CHECK(
                plan.old_pos[lo] % page_len == 0 && old_page_pos >= 0 &&
                    old_page_pos < static_cast<int64_t>(old_pages.size()),
                "recompact_with: survivor page not page-aligned"
            );
            new_pages[np] = old_pages[old_page_pos];
            for (int64_t i = lo; i < hi; ++i) {
                TORCH_CHECK(
                    plan.survivor[i] && plan.old_pos[i] == old_page_pos * page_len + (i - lo),
                    "recompact_with: survivor page not atomic (would need a "
                    "survivor copy)"
                );
            }
        } else {
            // Recalled page: claim a fresh physical page and record its K/V writes.
            // Per-page seed s = seed_density * bc(age) / length_gain lands the page's
            // bias-corrected max-over-query-head eviction density exactly at
            // seed_density, which is the page's OWN recall score and nothing else. A
            // uniform seed s in every (query-head, slot) reads back as s * length_gain
            // / bc; no group factor, mass is stored per query head, and length_gain is 1
            // unless a reference length is set (see evict). age = stream_len - base
            // address (plan.addr[lo]); bc = (1 - (1-decay)^age) floored at eps (decay ==
            // 0 or age <= 0 => bc == 1). seed_density <= 0 => s = 0 (page enters cold).
            const int32_t phys = this->claim_pages(1).front();
            new_pages[np] = phys;
            const int64_t age = stream_len - plan.addr[lo];
            const double bc = ema_bias_correction(decay, age);
            const double seed_density = plan.score[lo];
            const double s = seed_density > 0.0 ? seed_density * bc / this->density_length_gain() : 0.0;
            for (int64_t i = lo; i < hi; ++i) {
                TORCH_CHECK(
                    !plan.survivor[i],
                    "recompact_with: mixed survivor/recalled page (would need "
                    "a survivor copy)"
                );
                recalled_dst_slots.push_back(phys * static_cast<int32_t>(page_len) + static_cast<int32_t>(i - lo));
                recalled_rows.push_back(plan.recalled_row[i]);
                recalled_seed.push_back(static_cast<float>(s));
            }
        }
        for (int64_t i = lo; i < hi; ++i) {
            all_dst_slots[i] = new_pages[np] * static_cast<int32_t>(page_len) + static_cast<int32_t>(i - lo);
            // plan.old_rope is the token's BAKED position, not its slot index; its new
            // one comes from the merged buffer's layout at slot index i, which is the
            // index only when the buffer ropes contiguously.
            old_rope_i32[i] = static_cast<int32_t>(plan.old_rope[i]);
            new_pos_i32[i] = static_cast<int32_t>(this->rope_of(merged_layout, i));
        }
    }

    std::vector<int64_t> recalled_dst_i64(recalled_dst_slots.begin(), recalled_dst_slots.end());
    at::Tensor recalled_rows_dev = at::empty(
        {static_cast<int64_t>(recalled_rows.size())},
        at::TensorOptions().dtype(at::kLong)
    );
    if (!recalled_rows.empty()) {
        std::memcpy(
            recalled_rows_dev.data_ptr<int64_t>(),
            recalled_rows.data(),
            recalled_rows.size() * sizeof(int64_t)
        );
    }
    recalled_rows_dev = recalled_rows_dev.to(this->dev);
    at::Tensor recalled_dst_dev = at::empty(
        {static_cast<int64_t>(recalled_dst_slots.size())},
        at::TensorOptions().dtype(at::kInt)
    );
    if (!recalled_dst_slots.empty()) {
        std::memcpy(
            recalled_dst_dev.data_ptr<int32_t>(),
            recalled_dst_slots.data(),
            recalled_dst_slots.size() * sizeof(int32_t)
        );
    }
    recalled_dst_dev = recalled_dst_dev.to(this->dev);
    at::Tensor recalled_dst_i64_dev = at::empty(
        {static_cast<int64_t>(recalled_dst_i64.size())},
        at::TensorOptions().dtype(at::kLong)
    );
    if (!recalled_dst_i64.empty()) {
        std::memcpy(
            recalled_dst_i64_dev.data_ptr<int64_t>(),
            recalled_dst_i64.data(),
            recalled_dst_i64.size() * sizeof(int64_t)
        );
    }
    recalled_dst_i64_dev = recalled_dst_i64_dev.to(this->dev);
    at::Tensor all_dst_dev = at::empty(
        {static_cast<int64_t>(all_dst_slots.size())},
        at::TensorOptions().dtype(at::kInt)
    );
    if (!all_dst_slots.empty()) {
        std::memcpy(all_dst_dev.data_ptr<int32_t>(), all_dst_slots.data(), all_dst_slots.size() * sizeof(int32_t));
    }
    all_dst_dev = all_dst_dev.to(this->dev);
    at::Tensor old_rope_dev = at::empty(
        {static_cast<int64_t>(old_rope_i32.size())},
        at::TensorOptions().dtype(at::kInt)
    );
    if (!old_rope_i32.empty()) {
        std::memcpy(old_rope_dev.data_ptr<int32_t>(), old_rope_i32.data(), old_rope_i32.size() * sizeof(int32_t));
    }
    old_rope_dev = old_rope_dev.to(this->dev);
    at::Tensor new_pos_dev = at::empty({static_cast<int64_t>(new_pos_i32.size())}, at::TensorOptions().dtype(at::kInt));
    if (!new_pos_i32.empty()) {
        std::memcpy(new_pos_dev.data_ptr<int32_t>(), new_pos_i32.data(), new_pos_i32.size() * sizeof(int32_t));
    }
    new_pos_dev = new_pos_dev.to(this->dev);
    // Per recalled slot, its page's seed s broadcast across the n_q_heads columns of
    // mass_flat, for the per-(layer, query-head) mass overwrite below.
    at::Tensor seed_mat;
    if (!recalled_dst_slots.empty()) {
        seed_mat = at::tensor(recalled_seed, at::TensorOptions().dtype(at::kFloat))
                       .to(this->dev)
                       .unsqueeze(1)
                       .expand({static_cast<int64_t>(recalled_seed.size()), this->n_q_heads})
                       .contiguous();
    }

    for (int64_t l = 0; l < this->num_layers; ++l) {
        auto k_pool = this->k_pools.select(0, l);
        auto v_pool = this->v_pools.select(0, l);
        // Write recalled K/V into their fresh slots (CPU->GPU, the sole data copy).
        // write_kv takes [num, n_kv_heads, head_dim] on device. Overwrite each
        // recalled slot's per-(layer, query-head) mass with the page's seed s (same
        // across layers and query heads; overwrites, not accumulates, over any stale
        // mass).
        if (!recalled_dst_slots.empty()) {
            auto k_sel = recalled_k[l].index_select(0, recalled_rows_dev).to(this->dev, k_pool.scalar_type());
            auto v_sel = recalled_v[l].index_select(0, recalled_rows_dev).to(this->dev, v_pool.scalar_type());
            write_kv_cuda(k_pool, v_pool, k_sel, v_sel, recalled_dst_dev);
            auto mass_flat = this->mass_pools.select(0, l).view({this->page_count * page_len, this->n_q_heads});
            mass_flat.index_copy_(0, recalled_dst_i64_dev, seed_mat);
        }
        // Re-rope every token from its baked old pos to the merged layout's pos.
        // Survivor keys re-rope IN PLACE; delta-0 rows are kernel no-ops. V and
        // survivor mass keep their slots.
        reposition_kv_cuda(k_pool, all_dst_dev, old_rope_dev, new_pos_dev, this->rope_theta);
    }

    // Install the permuted table and set the new context length. No survivor pages
    // are freed.
    it->second = std::move(new_pages);
    this->lens[seq_id] = N;
    this->rope_layout[seq_id] = merged_layout;

    // Rebuild identity to the merged addr order the plan lays out. Each new page
    // holds one whole address-aligned page; the tail may be partial. page_id is the
    // page's address / page_size; the whole-page invariant is asserted per page.
    std::vector<int64_t> new_pidx(n_new_pages);
    std::vector<std::vector<int64_t>> new_tok(n_new_pages);
    int64_t max_pidx = -1;
    for (int64_t np = 0; np < n_new_pages; ++np) {
        const int64_t lo = np * page_len;
        const int64_t hi = std::min(lo + page_len, N);
        const int64_t base = plan.addr[lo];
        TORCH_CHECK(
            base % page_len == 0,
            "recompact_with: page not address-aligned (addr ",
            base,
            " not a multiple of page_size ",
            page_len,
            ")"
        );
        new_pidx[np] = base / page_len;
        new_tok[np].reserve(hi - lo);
        for (int64_t i = lo; i < hi; ++i) {
            TORCH_CHECK(plan.addr[i] == base + (i - lo), "recompact_with: page addresses not contiguous/aligned");
            new_tok[np].push_back(plan.tok[i]);
        }
        max_pidx = std::max(max_pidx, new_pidx[np]);
    }
    this->page_ids.at(seq_id) = std::move(new_pidx);
    this->tok.at(seq_id) = std::move(new_tok);
    this->next_pidx.at(seq_id) = std::max(this->next_pidx.at(seq_id), max_pidx + 1);
}

void ActiveBuffer::set_identity(
    int64_t seq_id,
    int64_t start_pos,
    const std::vector<int64_t>& addr,
    const std::vector<int64_t>& toks
) {
    TORCH_CHECK(this->tables.find(seq_id) != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(addr.size() == toks.size(), "set_identity: addr/token length mismatch");
    TORCH_CHECK(start_pos >= 0, "set_identity: start_pos must be >= 0");
    const int64_t page_len = this->page_len;
    auto& seq_pidx = this->page_ids.at(seq_id);
    auto& seq_tok = this->tok.at(seq_id);
    auto& seq_next = this->next_pidx.at(seq_id);
    const int64_t held = static_cast<int64_t>(this->tables.at(seq_id).size());
    for (size_t i = 0; i < addr.size(); ++i) {
        const int64_t p = start_pos + static_cast<int64_t>(i);
        const int64_t pp = p / page_len;
        const int64_t off = p % page_len;
        TORCH_CHECK(pp < held, "set_identity: page position ", pp, " beyond the ", held, " pages held by seq ", seq_id);
        const int64_t a = addr[i];
        // Whole-page address-aligned invariant: the token at page offset off holds
        // address page_id*page_size + off, so its page index is (a - off)/page_size and
        // a - off must be a page multiple.
        TORCH_CHECK(
            (a - off) % page_len == 0,
            "set_identity: address ",
            a,
            " at page offset ",
            off,
            " breaks the address-aligned invariant (page_size ",
            page_len,
            ")"
        );
        const int64_t page_idx = (a - off) / page_len;
        if (pp >= static_cast<int64_t>(seq_pidx.size())) {
            seq_pidx.resize(pp + 1, -1);
            seq_tok.resize(pp + 1);
        }
        if (seq_pidx[pp] < 0) {
            seq_pidx[pp] = page_idx;
        } else {
            TORCH_CHECK(
                seq_pidx[pp] == page_idx,
                "set_identity: page position ",
                pp,
                " address-crosses page ",
                seq_pidx[pp],
                " vs ",
                page_idx
            );
        }
        auto& ids = seq_tok[pp];
        if (off >= static_cast<int64_t>(ids.size())) {
            ids.resize(off + 1, -1);
        }
        ids[off] = toks[i];
        seq_next = std::max(seq_next, page_idx + 1);
    }
}

int64_t ActiveBuffer::num_active_pages(int64_t seq_id) const {
    auto it = this->page_ids.find(seq_id);
    TORCH_CHECK(it != this->page_ids.end(), "ActiveBuffer: unknown seq ", seq_id);
    return static_cast<int64_t>(it->second.size());
}

int64_t ActiveBuffer::page_id(int64_t seq_id, int64_t pp) const {
    auto it = this->page_ids.find(seq_id);
    TORCH_CHECK(it != this->page_ids.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        pp >= 0 && pp < static_cast<int64_t>(it->second.size()),
        "ActiveBuffer: page position ",
        pp,
        " out of range"
    );
    return it->second[pp];
}

const std::vector<int64_t>& ActiveBuffer::page_token_ids(int64_t seq_id, int64_t pp) const {
    auto it = this->tok.find(seq_id);
    TORCH_CHECK(it != this->tok.end(), "ActiveBuffer: unknown seq ", seq_id);
    TORCH_CHECK(
        pp >= 0 && pp < static_cast<int64_t>(it->second.size()),
        "ActiveBuffer: page position ",
        pp,
        " out of range"
    );
    return it->second[pp];
}

int64_t ActiveBuffer::next_page_index(int64_t seq_id) const {
    auto it = this->next_pidx.find(seq_id);
    TORCH_CHECK(it != this->next_pidx.end(), "ActiveBuffer: unknown seq ", seq_id);
    return it->second;
}

void ActiveBuffer::check_page_aligned(int64_t seq_id) const {
    auto pit = this->page_ids.find(seq_id);
    auto tit = this->tok.find(seq_id);
    TORCH_CHECK(pit != this->page_ids.end() && tit != this->tok.end(), "ActiveBuffer: unknown seq ", seq_id);
    const auto& seq_pidx = pit->second;
    const auto& seq_tok = tit->second;
    const int64_t page_len = this->page_len;
    TORCH_CHECK(seq_pidx.size() == seq_tok.size(), "check_page_aligned: page_id/tok size mismatch");
    const int64_t n = static_cast<int64_t>(seq_pidx.size());
    for (int64_t pp = 0; pp < n; ++pp) {
        TORCH_CHECK(seq_pidx[pp] >= 0, "check_page_aligned: page position ", pp, " has no recorded page index");
        if (pp > 0) {
            TORCH_CHECK(
                seq_pidx[pp] > seq_pidx[pp - 1],
                "check_page_aligned: page indices not strictly ascending at ",
                pp
            );
        }
        const int64_t len = static_cast<int64_t>(seq_tok[pp].size());
        TORCH_CHECK(
            len > 0 && len <= page_len,
            "check_page_aligned: page position ",
            pp,
            " holds ",
            len,
            " ids (expected 1..",
            page_len,
            ")"
        );
        // Only the last page position may be partial.
        if (pp + 1 < n) {
            TORCH_CHECK(
                len == page_len,
                "check_page_aligned: interior page position ",
                pp,
                " is partial (",
                len,
                " < ",
                page_len,
                ")"
            );
        }
        for (int64_t off = 0; off < len; ++off) {
            TORCH_CHECK(
                seq_tok[pp][off] >= 0,
                "check_page_aligned: page position ",
                pp,
                " offset ",
                off,
                " has no recorded token id"
            );
        }
    }
}

void ActiveBuffer::decay_mass(int64_t seq_id, double factor) {
    auto it = this->tables.find(seq_id);
    TORCH_CHECK(it != this->tables.end(), "ActiveBuffer: unknown seq ", seq_id);
    if (factor == 1.0) {
        return;
    }

    const int64_t ctx = this->lens[seq_id];
    const auto& pages = it->second;
    std::vector<int64_t> slots;
    slots.reserve(ctx);
    for (int64_t j = 0; j < ctx; ++j) {
        slots.push_back(pages[j / this->page_len] * this->page_len + (j % this->page_len));
    }
    at::Tensor slots_dev = at::empty({static_cast<int64_t>(slots.size())}, at::TensorOptions().dtype(at::kLong));
    if (!slots.empty()) {
        std::memcpy(slots_dev.data_ptr<int64_t>(), slots.data(), slots.size() * sizeof(int64_t));
    }
    slots_dev = slots_dev.to(this->dev);

    // Scale these slots' mass in every layer/query head; mass_pools is
    // [n_layers, num_pages, page_size, n_q_heads]. gather -> mul -> scatter.
    for (int64_t l = 0; l < this->num_layers; ++l) {
        auto mass_flat = this->mass_pools.select(0, l).view({this->page_count * this->page_len, this->n_q_heads});
        auto scaled = mass_flat.index_select(0, slots_dev).mul_(factor);
        mass_flat.index_copy_(0, slots_dev, scaled);
    }
}

void ActiveBuffer::upkeep(int64_t seq_id, int64_t n_tokens, double attention_mass_decay) {
    if (attention_mass_decay != 0.0 && n_tokens > 0) {
        this->decay_mass(seq_id, std::pow(1.0 - attention_mass_decay, static_cast<double>(n_tokens)));
    }
}

}  // namespace pulsar
