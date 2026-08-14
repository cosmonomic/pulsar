#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"  // ActiveBuffer, DemotedKV
#include "pulsar/runtime/kv/page_view.hpp"
#include "pulsar/runtime/kv/paged_pool.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// FileBucket: a spill bucket backed by disk. A per-sequence KV store whose page KV
// lives in a file while its per-page identity (page_id, token_ids, mass) stays in
// RAM. The bucket above spills its coldest pages here when it is over budget,
// token_ids moving WITH them (BM25 retrieval and the history merge never touch
// disk). Only a candidate's relevance-layer K (gather_k) or a promoted page's full
// KV (take_pages) is read back, bounded to the pages the cycle asked for.
//
// Per page the RAM index keeps {page_id, token_ids, mass, mass_step} plus a
// (offset, nbytes) slot in the sequence's disk file; addr/pos/segment_id are
// derivable and not stored. Disk layout: one file per sequence at
// <spill_dir>/<seq>, each spilled page's KV appended contiguously as all layers' K
// then all layers' V (this->dtype bytes). The file is created on begin() and
// deleted on end(). Engine-internal; not TorchBind-exposed.
//
// Satisfies the same SpillBucket surface as PagedBucket, looping internally where
// the API is batched.

namespace pulsar {

struct BucketFile;  // one sequence's file + RAM index; defined in file_bucket.cpp

struct FileBucket {
    // spill_dir is the parent directory for the per-sequence files; it is created if
    // absent. dtype is the on-disk (and pool) element type.
    FileBucket(
        int64_t n_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        at::ScalarType dtype,
        std::string spill_dir
    );
    ~FileBucket();
    // Movable, not copyable.
    FileBucket(FileBucket&&) noexcept;
    FileBucket& operator=(FileBucket&&) noexcept;

    // Disabled when spill_dir is empty: begin/end no-op and no file is ever created.
    bool enabled() const {
        return !this->spill_dir.empty();
    }

    void begin(int64_t seq);  // create/truncate <spill_dir>/<seq>, open its fd
    void end(int64_t seq);  // close the fd, unlink the file, drop the RAM index

    // The sequence's pages as a page-index cursor, ascending. token_ids stay in
    // RAM, so this cursor never touches disk.
    BucketView view(int64_t seq) const;

    // Host-only half of an accept: record each victim page's RAM index entry and
    // claim its byte range in the file, no I/O.
    Reservation reserve(int64_t seq, const DemotedKV& demoted, int64_t step);
    // The copy half: gather the reserved pages' K/V out of the ActiveBuffer's pool
    // and write them at their claimed offsets.
    void fill(int64_t seq, const ActiveBuffer& active, const Reservation& r);
    // Take a page from the bucket above (a spill): append its KV bytes and record
    // the RAM index entry. k/v are page-major [cnt, n_layers, n_kv_heads, head_dim];
    // they are moved to CPU/this->dtype and written as all layers' K then all layers'
    // V.
    void accept(int64_t seq, const PageKV& page);

    // The given held FULL pages' stored K on CPU; see GatheredK. Reads only the
    // requested layers' K blocks, not the whole page, straight into their destination
    // rows, so the order is the identity. Pages must be held.
    GatheredK gather_k(int64_t seq, const std::vector<int64_t>& page_ids, const std::vector<int64_t>& layers) const;

    // Promote: read the given pages' full KV back from disk and drop them from the
    // RAM index, in page_ids order, as page-major [cnt, n_layers, n_kv_heads,
    // head_dim] carriers. The disk bytes stay (space is reclaimed only when the whole
    // file is deleted on end()).
    std::vector<PageKV> take_pages(int64_t seq, const std::vector<int64_t>& page_ids);

    // Remove and return the coldest pages until the sequence holds <=
    // budget_tokens, by lazy-decayed effective mass (see PagedBucket). This is the
    // last bucket in the engine's chain, so nothing calls it there; it exists so
    // either position can hold either implementation.
    std::vector<PageKV> spill_to_budget(int64_t seq, int64_t budget_tokens, int64_t step, double decay);

    // End-of-cycle deferred release. A file bucket reclaims space only at end(), so
    // an erased page frees nothing and this is a no-op.
    void release_pending(int64_t seq);

    // Read a page's KV back from disk into CPU K/V [cnt, n_layers, n_kv_heads,
    // head_dim] via a pageable one-page buffer freed straight after. Does NOT
    // remove the page. Throws on an unknown page.
    void read_kv(int64_t seq, int64_t page_id, at::Tensor& k_out, at::Tensor& v_out) const;

    bool has_seq(int64_t seq) const;
    bool has_page(int64_t seq, int64_t page_id) const;

    int64_t bucket_tokens(int64_t seq) const;  // demoted tokens on disk for a seq
    int64_t size() const;  // total across sequences

  private:
    BucketFile& file(int64_t seq);
    const BucketFile& file(int64_t seq) const;
    int64_t page_nbytes(int64_t cnt) const;  // KV bytes for a cnt-token page
    // Append-claim a byte range for a cnt-token page and record its RAM index entry.
    int64_t claim(BucketFile& f, int64_t page_id, std::vector<int64_t> token_ids, double mass, int64_t mass_step);
    // Write one page's K then V at a claimed offset, each LAYER-MAJOR
    // [n_layers, cnt, n_kv_heads, head_dim] (the on-disk layout), from any device.
    void write_page(BucketFile& f, int64_t offset, const at::Tensor& k, const at::Tensor& v);

    int64_t n_layers;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t page_size;
    at::ScalarType dtype;
    std::string spill_dir;
    std::unordered_map<int64_t, std::unique_ptr<BucketFile>> seqs;
};

}  // namespace pulsar
