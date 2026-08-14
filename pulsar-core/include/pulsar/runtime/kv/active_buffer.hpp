#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// The attended-KV page pool the paged attention ops read, plus the eviction and recall
// carriers that move pages in and out of it. Engine-internal plain C++, not
// TorchBind-exposed; unit tests in pulsar-core/tests/runtime/.
//
// One pool per transformer layer. Layout:
//   k_pools, v_pools : [n_layers, num_pages, page_size, n_kv_heads, head_dim]
//   mass_pools       : [n_layers, num_pages, page_size, n_q_heads]  (fp32, per query head)
//   k_pool(layer)/v_pool(layer) : [num_pages, page_size, n_kv_heads, head_dim]
//   mass_pool(layer)            : [num_pages, page_size, n_q_heads]
//   slot = page_id * page_size + offset.
// Page tables, active lengths, and the free list are SHARED across layers (one
// physical page id set per sequence, valid in every layer's pool).

namespace pulsar {

// How a sequence's slot indices map to RoPE positions.
//   contiguous: the slot index IS the position.
//   compacted:  the three-region layout, whose distant region collapses onto one
//               position (see ActiveBuffer::rope_pos_at).
enum class PositionLayout { contiguous, compacted };

inline PositionLayout parse_position_layout(const std::string& name) {
    if (name == "contiguous") {
        return PositionLayout::contiguous;
    }
    if (name == "compacted") {
        return PositionLayout::compacted;
    }
    TORCH_CHECK(
        false,
        "active_position_layout: expected \"contiguous\" or \"compacted\", "
        "got \"",
        name,
        "\""
    );
}

// The tokens an evict() call removed, as SLOT references into the active pool (not a
// K/V carrier). K is re-roped in place to the session's short_offset at eviction, so
// the referenced slots hold keys at that one position; V is pos-invariant.
// victim_slots index the flat pool [num_pages*page_size, n_kv_heads, head_dim] in
// victim (addr-ascending whole-page) order; the first spill bucket's fill does the
// single copy that moves them out (a D2H). The slots are freed by demote_victim_pages
// but their K/V survives the free (free returns page ids only), so the fill must
// complete before the next claim_pages reuses them. addr()/tok() carry each evicted
// token's conversation address and token id, read from the active buffer's identity
// before the victim pages drop. M == 0 (empty tensors) when nothing was evicted.
struct DemotedKV {
    at::Tensor addr_data;  // int64 [M] on CPU, evicted addresses
    at::Tensor tok_data;  // int64 [M] on CPU, evicted token ids
    at::Tensor mass_data;  // fp32 [P] on CPU, per demoted PAGE relevance mass
    at::Tensor victim_slots;  // int64 [M] on device, flat-pool slot per evicted token

    at::Tensor addr() const {
        return this->addr_data;
    }
    at::Tensor tok() const {
        return this->tok_data;
    }
    // Per-page eviction-time mass (page order == addr ascending). A page whose mass
    // the caller did not record enters as 0.
    at::Tensor page_mass() const {
        return this->mass_data;
    }
    int64_t num_tokens() const {
        return this->addr_data.numel();
    }
};

// Read-only probe over an eviction's per-page density, bucketed by NORMALIZED page
// position across ALL of the sequence's pages in address order, the sink prefix and
// the working window included.
//
// raw_mass_sum is the per-page attention mass BEFORE the age bias correction,
// corrected_density_sum the same page AFTER it, so the two differ only by that divide;
// both are captured before the neighbourhood max. Sums, not means: page_count carries the pages
// behind each bucket and the consumer divides. evict ADDS into an instance across
// calls.
struct DensityByPosition {
    static constexpr int64_t buckets = 16;

    std::array<double, buckets> raw_mass_sum{};
    std::array<double, buckets> corrected_density_sum{};
    std::array<int64_t, buckets> page_count{};

    // num_pages must be > 0.
    static int64_t bucket_of(int64_t page_index, int64_t num_pages) {
        return std::min(buckets - 1, page_index * buckets / num_pages);
    }
};

// Why one evict() call demoted what it did, and what a selection with no age
// correction would have demoted instead. Counters only: nothing here feeds back into
// which pages are demoted. Filled per call (NOT accumulated), and left untouched by a
// call with no evictable candidate.
//
// bar_pages + backstop_pages is the call's whole victim set: bar_pages cleared the
// keep-bar test, backstop_pages were taken afterwards to reach active_cap.
//
// candidate_age/victim_age are ages in TOKENS, stream_len minus the page's base
// address, over all candidates and over the victims alone.
//
// raw_mass_victims/raw_mass_agree are the counterfactual: rank the same candidates by
// their neighbourhood RAW mass (no age correction) and take raw_mass_victims of them,
// the count the real selection took; raw_mass_agree is how many of those the real
// victim set also holds.
struct EvictAttribution {
    int64_t bar_pages = 0;
    int64_t backstop_pages = 0;
    std::vector<int64_t> candidate_age;
    std::vector<int64_t> victim_age;
    int64_t raw_mass_victims = 0;
    int64_t raw_mass_agree = 0;
};

// A recall's recompact plan: N tokens in address (temporal) order, one entry per
// new slot index i. survivor[i] != 0: a surviving active token at old slot index
// old_pos[i]. survivor[i] == 0: a recalled token with K/V at recalled_row[i].
// old_rope[i] is the token's baked roped position: the layout's position for a
// survivor's old slot index, the demotion position (the session's short_offset) for a
// recalled token. The two coincide only for a survivor under a contiguous outgoing
// layout; only old_pos is a buffer coordinate, and only it carries the page alignment
// recompact_with checks. addr[i]/tok[i] are the token's conversation address and token
// id. Consumed by ActiveBuffer::recompact_with. All seven vectors share length N.
struct RecompactPlan {
    std::vector<char> survivor;
    std::vector<int64_t> old_pos;  // old slot index (valid where survivor)
    std::vector<int64_t> old_rope;  // baked roped pos of each token
    std::vector<int64_t> recalled_row;  // recalled-set row (valid where !survivor)
    std::vector<int64_t> addr;  // conversation address of each token
    std::vector<int64_t> tok;  // token id of each token
    // Recall score of each recalled token's page (valid where !survivor); the seed
    // density recompact_with lands the page at.
    std::vector<double> score;
};

// Owns the attended KV page pool and hands out physical pages to sequences. Tracks a
// per-sequence page table (physical page ids) and active length (attended tokens whose
// KV lives in the pool).
//
// Also owns per-sequence page-index/token-id identity, in page-position order
// parallel to the page table:
//   page_ids[pp] -> the immutable page index (address / page_size) at page
//     position pp. Assigned once from set_identity's token addresses, preserved
//     verbatim through evict/recompact_with. Ascending in pp, gappy in index
//     space (evicted indices missing).
//   tok[pp] -> the token ids held at page position pp (page_size ids; fewer
//     for the conversation tail page).
// Whole-page address-aligned invariant: page position pp holds exactly addresses
// [page_ids[pp]*page_size, page_ids[pp]*page_size + tok[pp].size()), the token at
// page offset o carrying address page_ids[pp]*page_size + o. set_identity and
// recompact_with assert it; check_page_aligned re-checks a whole sequence.
struct ActiveBuffer {
    // device: e.g. "cuda", "cuda:0", "cpu".
    // n_layers pools allocated; page tables + active_len + free-list shared
    // across layers (same physical page ids in every layer's pool).
    // n_q_heads (>= n_kv_heads, a multiple of it) sizes the per-query-head mass
    // pool the density max ranges over; the group is n_q_heads / n_kv_heads.
    // rope_theta: RoPE base keys are roped with; evict re-ropes survivors with it.
    ActiveBuffer(
        int64_t n_layers,
        int64_t num_pages,
        int64_t page_size,
        int64_t n_kv_heads,
        int64_t n_q_heads,
        int64_t head_dim,
        at::ScalarType dtype,
        std::string device,
        double rope_theta
    );

    // ceil(num_tokens / page_size) pages available on the free list?
    bool can_allocate(int64_t num_tokens) const;

    // Reserve pages for a NEW sequence of num_tokens; records its page table
    // and sets active_len = num_tokens. Throws if seq_id already exists or OOM.
    void allocate(int64_t seq_id, int64_t num_tokens);

    // Grow an existing sequence by num_new_tokens, claiming pages across page
    // boundaries as needed; advances active_len. Throws on OOM.
    void append(int64_t seq_id, int64_t num_new_tokens);

    // New pages a subsequent append(seq_id, num_new_tokens) would claim.
    int64_t pages_to_append(int64_t seq_id, int64_t num_new_tokens) const;

    // Return the sequence's pages to the free list and forget it.
    void free(int64_t seq_id);

    at::Tensor page_table(int64_t seq_id) const;  // int32 [num_pages_held]
    int64_t active_len(int64_t seq_id) const;
    bool has_seq(int64_t seq_id) const;

    // Physical slots for num_tokens appended at context pos
    // [start_pos, start_pos + num_tokens), for write_kv. int32.
    at::Tensor slot_mapping(int64_t seq_id, int64_t start_pos, int64_t num_tokens) const;

    // Page-level density eviction with sink/working protection against an ABSOLUTE
    // keep-bar. Evictable = a FULL page inside the band
    // [min(n_sink, ctx), max(0, ctx - n_working)): the sink prefix [0, n_sink) and
    // working suffix [ctx - n_working, ctx) are never candidates. Demote each evictable
    // page whose density < min_density.
    // n_sink/n_working/layout/short_offset are the CALLING SEQUENCE's and lay out the
    // survivors' RoPE positions (see rope_pos_at). short_offset is an absolute position
    // in TOKENS, must be > 0, and is the position demoted keys are roped to under BOTH
    // layouts.
    // Density: over the per-query-head mass, MEAN over ALL query heads, then MAX
    // over the page's slots, then MAX over the relevance layers (see
    // set_relevance_layers), divided by the bias-correction denominator
    // bc = (1 - (1-alpha)^age) for alpha = attention_mass_decay, the calling
    // sequence's EMA rate (age = stream_len - the page's absolute
    // address; age <= 0 or alpha <= 0 => bc == 1, floored at 1e-6). The reduction before
    // the divide is an attention share as a multiple of uniform over the length the
    // mass is stated against, so 1.0 is parity at THAT length and nothing bounds it
    // above. The length is each query's own attended key count unless
    // set_mass_reference_length names one, which is then multiplied in here rather than
    // by the kernels. No group divisor: mass is stored per query head, so the reduction
    // already is a per-forward softmax fraction. The slot and layer axes are always max.
    // The head axis collapses BEFORE the slot max, so the density is one token's score,
    // not a blend of per-head favourites. The recall reranker takes the same mean over
    // query heads, so both directions measure the same quantity under one derived bar.
    // block_radius, in PAGES, > 0 replaces each page's density with the MAX over the
    // window 2*radius+1 before the threshold, so a page survives exactly when some page
    // within the radius clears the bar. There is no block structure: every page carries
    // its own neighbourhood's max.
    // Backstop: while active_len exceeds active_cap, demote lowest-density pages toward
    // it (<= 0 disables), ties broken by age (page_id ascending, oldest first) since a
    // neighbourhood shares one density. Bounded by the evictable band, so active_cap is
    // a target, not a guarantee.
    // Survivors keep their physical pages; their slot indices close up to
    // [0, new_len) in original order, and each survivor whose position moves under the
    // new layout has its K re-roped in place (reposition_kv) old->new pos in every
    // layer. V and mass are pos-independent, not moved. Victim pages are freed and the
    // page table rewritten to the survivors in original order. Returns a DemotedKV: the
    // victims' pool slots plus their addresses, token ids and per-page mass. Empty
    // DemotedKV (M == 0) when nothing is evicted.
    // cand_density, when non-null, is filled (in candidate order) with each candidate's
    // neighbourhood bias-corrected density: the value min_density compares against.
    // position_profile, when non-null, is ACCUMULATED into (see DensityByPosition).
    // attribution, when non-null, is OVERWRITTEN with this call's cause split (see
    // EvictAttribution).
    // No out-param affects which pages are demoted. A call with no evictable
    // candidate returns before any of them is touched.
    DemotedKV evict(
        int64_t seq_id,
        int64_t n_sink,
        int64_t n_working,
        PositionLayout layout,
        int64_t short_offset,
        double min_density,
        int64_t active_cap,
        int64_t block_radius,
        int64_t stream_len,
        double attention_mass_decay = 0.0,
        std::vector<float>* cand_density = nullptr,
        DensityByPosition* position_profile = nullptr,
        EvictAttribution* attribution = nullptr
    );

    // The layer subset evict reduces the per-page mass over. Must be in range and a
    // CONTIGUOUS ascending run: evict narrows mass_pools to that layer range, and
    // narrow only stays a view (no copy of the pool) when the layers are adjacent.
    // Empty states that nothing evicts, and then evict must not be called. Defaults
    // (never called) to ALL layers [0, num_layers).
    void set_relevance_layers(std::vector<int64_t> layers);

    // The length in TOKENS the eviction density is stated against. A key holding a
    // uniform share of a context this long reads back at exactly 1. <= 0 (the default)
    // states the density against whatever length the attention kernels already applied,
    // which is each query's own attended key count. A configured length is a CONSTANT,
    // so the kernels accumulate the plain weight and evict restores it once per page;
    // recompact_with divides it back out of the mass it seeds. The caller must run the
    // attention kernels at mass_length_gain 1 whenever it sets one, or the factor is
    // applied twice.
    void set_mass_reference_length(int64_t tokens) {
        this->mass_reference_length = tokens;
    }

    // Whether the attention kernels must accumulate this layer's mass side output.
    // evict is mass's only reader and reduces over the relevance layers alone, so
    // every other layer's mass is written and never read.
    bool mass_is_read(int64_t layer) const {
        return layer >= this->relevance_offset && layer < this->relevance_offset + this->relevance_count;
    }

    // Recompute the sequence's RoPE layout for its CURRENT active length, re-rope every
    // token whose position moves under it, and install it. Nothing else changes: the
    // page table, active_len, identity, V and mass are all untouched.
    //
    // A caller that lets the buffer grow must run this whether or not anything was
    // evicted or recalled: growth alone slides the working run forward and leaves the
    // tokens behind it belonging at short_offset.
    //
    // old_index[i] is the slot index the token now at i held under the INSTALLED
    // (outgoing) layout; its baked position is read there. Empty is the identity.
    // Keys re-rope on the POSITION moving, not the slot index: a token whose position is
    // unchanged is not touched.
    void relayout(
        int64_t seq_id,
        int64_t n_sink,
        int64_t n_working,
        PositionLayout layout,
        int64_t short_offset,
        const std::vector<int64_t>& old_index = {}
    );

    // Re-lay a sequence's surviving active tokens and externally recalled tokens into
    // slot indices [0, N) in the plan's addr order by PERMUTING the page table,
    // re-roping every shifted key in place. Pages are ATOMIC: no survivor K/V is
    // copied. A survivor page keeps its tokens in their physical slots and only its
    // slot index, and so its position under the merged layout, changes; each recalled
    // page is written from recalled_k/v (per layer [R, n_kv_heads, head_dim], any
    // device/dtype, in recalled-row order) into a freshly claimed page (the sole data
    // copy, CPU->GPU).
    // The plan must be page-aligned in plan.old_pos, the SLOT-INDEX coordinate: each
    // block of pl new slot indices is one whole survivor page (offsets preserved) or one
    // recalled page, only the last partial (the survivor tail).
    // n_sink/n_working/layout/short_offset are the calling sequence's and lay out the
    // merged buffer's RoPE positions (see rope_pos_at); each token's baked position
    // comes from plan.old_rope, which is not plan.old_pos unless the outgoing layout is
    // contiguous.
    // active_len is set to N. K is re-roped (reposition_kv); V and
    // survivor mass are left in place.
    // Each recalled page's slots are seeded (every layer + query head) with
    // s = plan.score[page] * bc(age) / the reference length's gain (1 unless
    // set_mass_reference_length names one), landing the page's bias-corrected
    // mean-over-query-head eviction density at the page's recall score:
    // age = stream_len - the page's base address,
    // bc = (1 - (1-alpha)^age) for alpha = attention_mass_decay, the calling sequence's
    // EMA rate (age <= 0 or alpha <= 0 => bc == 1, floored at 1e-6). No group factor: a
    // uniform per-query-head seed s reads back as s * that gain / bc. A score <= 0 seeds
    // zero mass.
    void recompact_with(
        int64_t seq_id,
        int64_t n_sink,
        int64_t n_working,
        PositionLayout layout,
        int64_t short_offset,
        const RecompactPlan& plan,
        const std::vector<at::Tensor>& recalled_k,
        const std::vector<at::Tensor>& recalled_v,
        int64_t stream_len = 0,
        double attention_mass_decay = 0.0
    );

    // Multiply a sequence's active mass (every layer/head of its slots) by
    // factor, in place. factor == 1.0 is a no-op.
    void decay_mass(int64_t seq_id, double factor);

    // One EMA upkeep step, called once per forward with that forward's token count and
    // BEFORE the attention kernels add the mass it received, so mass tracks
    // m_new = (1-alpha)^n_tokens * m_old + received. alpha = attention_mass_decay is
    // the calling sequence's PER-TOKEN forgetting rate, so the EMA window is measured
    // in tokens; 0.0 leaves the mass untouched.
    void upkeep(int64_t seq_id, int64_t n_tokens, double attention_mass_decay);

    // Per-layer pool views (layer in [0, n_layers)), shaped exactly as the paged
    // ops take: k/v [num_pages, page_size, n_kv_heads, head_dim], mass
    // [num_pages, page_size, n_q_heads].
    at::Tensor k_pool(int64_t layer) const {
        return this->k_pools.select(0, layer);
    }
    at::Tensor v_pool(int64_t layer) const {
        return this->v_pools.select(0, layer);
    }
    at::Tensor mass_pool(int64_t layer) const {
        return this->mass_pools.select(0, layer);
    }

    // Full stacked pools; k_pool(layer) is a .select(0, layer) view into these.
    at::Tensor k_pool_all() const {
        return this->k_pools;
    }
    at::Tensor v_pool_all() const {
        return this->v_pools;
    }
    at::Tensor mass_pool_all() const {
        return this->mass_pools;
    }

    int64_t n_layers() const {
        return this->num_layers;
    }
    at::Device device() const {
        return this->dev;
    }

    int64_t num_free_pages() const {
        return static_cast<int64_t>(this->free_list.size());
    }
    int64_t num_pages() const {
        return this->page_count;
    }
    int64_t page_size() const {
        return this->page_len;
    }

    // Physical page ids held by one sequence; exposed for tests.
    const std::vector<int32_t>& pages_of(int64_t seq_id) const;

    // Record identity for tokens that became active at contiguous pool pos
    // [start_pos, start_pos + num_tokens): addr[i]/toks[i] are the conversation
    // address and token id of the token at start_pos + i. Grows/fills
    // page_ids[]/tok[] over the touched pages and advances next_page_index. Asserts
    // the whole-page address-aligned invariant per touched page.
    void
    set_identity(int64_t seq_id, int64_t start_pos, const std::vector<int64_t>& addr, const std::vector<int64_t>& toks);

    // The RoPE position baked into the token at slot index `index`, under the layout
    // this sequence's last relayout/recompact installed, and, at index == active_len,
    // the position the next appended token takes.
    //
    // A sequence under PositionLayout::contiguous gets the identity: slot index IS the
    // RoPE position. Under PositionLayout::compacted the buffer carries three regions,
    // laid out at each relayout/recompact from the calling sequence's knobs and extended
    // by appends in between:
    //
    //   sink     index < n_sink                  -> index        (its own prefix)
    //   distant  n_sink <= index < working_lo    -> short_offset (ALL one position)
    //   working  index >= working_lo             -> short_offset + 1 + index - working_lo
    //
    // working_lo is working_start(ctx, n_sink, n_working) as of the last layout. Tokens
    // appended since then continue the working run, so the position stays bounded by
    // short_offset + n_working + (tokens appended since the last relayout) however long
    // the conversation runs. A sequence that has not been through a layout yet is
    // contiguous.
    //
    // Slot indices are NOT positions under this layout: slot_mapping, seqlens_k,
    // context_lens and set_identity all keep taking the index, and only RoPE takes this.
    int64_t rope_pos_at(int64_t seq_id, int64_t index) const;

    // The slot index the protected working run starts at in a buffer of `ctx` tokens
    // under these knobs. Under the compacted layout it is where the working region's
    // positions begin.
    static int64_t working_start(int64_t ctx, int64_t n_sink, int64_t n_working) {
        return std::max(std::min(n_sink, ctx), ctx - n_working);
    }

    int64_t num_active_pages(int64_t seq_id) const;  // page positions held
    int64_t page_id(int64_t seq_id, int64_t pp) const;  // page_ids[pp]
    const std::vector<int64_t>& page_token_ids(int64_t seq_id,
                                               int64_t pp) const;  // tok[pp]

    // Pages created so far for the sequence (max address / page_size + 1). Upper
    // bound: the active page ids united with the demoted ones cover
    // [0, next_page_index).
    int64_t next_page_index(int64_t seq_id) const;
    // Re-assert the whole-page address-aligned invariant across the whole
    // sequence: page_ids recorded and strictly ascending, each page's ids sized
    // 1..page_size (only the last may be partial), and every token id recorded.
    void check_page_aligned(int64_t seq_id) const;

  private:
    std::vector<int32_t> claim_pages(int64_t n);  // pop + zero their mass (all layers)

    // Shared demote core for evict(): given a victim mask over the sequence's logical
    // pages (is_victim[j] != 0 => demote page j) and each page's ranking mass
    // (mass_of[j], 0 when absent), rope the victim K to short_offset in place and record
    // its slots as a DemotedKV, re-rope shifted survivors in place, free the victim
    // pages, and rebuild identity to the survivors. An all-zero mask returns an empty
    // DemotedKV. Every victim must be a full page.
    DemotedKV demote_victim_pages(
        int64_t seq_id,
        const std::vector<char>& is_victim,
        const std::unordered_map<int64_t, float>& mass_of,
        int64_t n_sink,
        int64_t n_working,
        PositionLayout layout,
        int64_t short_offset
    );

    // Per-sequence RoPE layout: which mapping is in force, the position the distant
    // region collapses onto, where the working run starts, and the sink width it was
    // laid out against. Recorded per sequence, installed by that sequence's cycle. A
    // contiguous layout, or working_lo < 0 (no cycle yet), is the identity mapping.
    struct RopeLayout {
        PositionLayout position_layout = PositionLayout::contiguous;
        int64_t working_lo = -1;
        int64_t n_sink = 0;
        int64_t short_offset = 0;
    };
    // The layout's position for a slot index, evaluated against an EXPLICIT layout so a
    // demote/recompact can hold the outgoing and incoming layouts side by side and
    // re-rope exactly the tokens whose position moves.
    int64_t rope_of(const RopeLayout& layout, int64_t index) const;
    // The layout a buffer of `ctx` tokens takes under the calling sequence's knobs;
    // installed at the end of each cycle.
    RopeLayout
    layout_for(int64_t ctx, int64_t n_sink, int64_t n_working, PositionLayout layout, int64_t short_offset) const;
    // The sequence's currently installed layout; contiguous for an unknown sequence.
    RopeLayout layout_of(int64_t seq_id) const;

    std::unordered_map<int64_t, RopeLayout> rope_layout;

    int64_t num_layers;
    int64_t page_count;
    int64_t page_len;
    int64_t n_kv_heads;
    int64_t n_q_heads;
    int64_t head_dim;
    double rope_theta;
    at::Tensor k_pools;  // [n_layers, num_pages, page_size, n_kv_heads, head_dim]
    at::Tensor v_pools;  // same shape as k_pools
    at::Tensor mass_pools;  // [n_layers, num_pages, page_size, n_q_heads] fp32 (per query head)
    at::Device dev;

    // Relevance layers evict takes the per-page mass MAX over: the contiguous run
    // [relevance_offset, relevance_offset + relevance_count) (all layers by default).
    int64_t relevance_offset;
    int64_t relevance_count;

    // See set_mass_reference_length. 0 leaves the kernels' own length in force.
    int64_t mass_reference_length = 0;
    // The constant the stored mass is short of the density's units: the reference
    // length, or 1 when the kernels already applied their own.
    double density_length_gain() const {
        return this->mass_reference_length > 0 ? static_cast<double>(this->mass_reference_length) : 1.0;
    }

    std::vector<int32_t> free_list;  // free physical page ids (stack)
    std::unordered_map<int64_t, std::vector<int32_t>> tables;  // seq -> pages
    std::unordered_map<int64_t, int64_t> lens;  // seq -> active_len

    // Token identity, page-position order parallel to `tables` (see class doc).
    std::unordered_map<int64_t, std::vector<int64_t>> page_ids;  // seq -> page id per pp
    std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> tok;  // seq -> token ids per pp
    std::unordered_map<int64_t, int64_t> next_pidx;  // seq -> pages created so far
};

}  // namespace pulsar
