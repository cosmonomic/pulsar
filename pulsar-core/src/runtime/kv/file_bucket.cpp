#include "pulsar/runtime/kv/file_bucket.hpp"

#include "host_pages.hpp"

#include <ATen/ATen.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <utility>

// FileBucket: a disk-backed spill bucket. One file per sequence at
// <spill_dir>/<seq>; each spilled page's KV is appended contiguously as all layers'
// K then all layers' V (this bucket's dtype), each block in POOL layout
// [cnt, n_kv_heads, head_dim] so a pread destination is directly usable and nothing
// is restaged. A RAM index keeps {page_id, token_ids, mass, mass_step, (offset,
// nbytes)} per page, sorted ascending by page index. Reads use pread into a pageable
// host buffer. The whole file is deleted on end(); dropping a page only removes its
// RAM index entry, so disk space is reclaimed at end, not per promotion.

namespace pulsar {

// One bucket page's RAM index entry: identity + token_ids + the eviction-time mass
// snapshot in RAM, KV bytes at [offset, offset + nbytes) in the sequence's file.
struct BucketPage {
    int64_t page_id;
    std::vector<int64_t> token_ids;
    int64_t offset;
    int64_t nbytes;
    double mass = 0.0;
    int64_t mass_step = 0;
};

// One sequence's disk file + page-index-sorted RAM index + append cursor.
struct BucketFile {
    std::string path;
    int fd = -1;
    int64_t write_offset = 0;
    std::vector<BucketPage> pages;  // ascending by page_id

    // Index of a page by page_id, or -1.
    int64_t find(int64_t page_id) const {
        auto it = std::lower_bound(
            this->pages.begin(),
            this->pages.end(),
            page_id,
            [](const BucketPage& p, int64_t idx) { return p.page_id < idx; }
        );
        if (it == this->pages.end() || it->page_id != page_id) {
            return -1;
        }
        return static_cast<int64_t>(it - this->pages.begin());
    }
};

FileBucket::FileBucket(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    at::ScalarType dtype,
    std::string spill_dir
)
    : n_layers(n_layers),
      n_kv_heads(n_kv_heads),
      head_dim(head_dim),
      page_size(page_size),
      dtype(dtype),
      spill_dir(std::move(spill_dir)) {
    if (this->enabled()) {
        std::filesystem::create_directories(this->spill_dir);
    }
}

FileBucket::~FileBucket() {
    for (auto& [seq, f] : this->seqs) {
        if (f && f->fd >= 0) {
            ::close(f->fd);
            std::error_code ec;
            std::filesystem::remove(f->path, ec);
        }
    }
}

FileBucket::FileBucket(FileBucket&&) noexcept = default;
FileBucket& FileBucket::operator=(FileBucket&&) noexcept = default;

int64_t FileBucket::page_nbytes(int64_t cnt) const {
    const int64_t elt = at::elementSize(this->dtype);
    // all layers' K then all layers' V; each block [cnt, n_kv_heads, head_dim].
    return this->n_layers * 2 * this->n_kv_heads * cnt * this->head_dim * elt;
}

BucketFile& FileBucket::file(int64_t seq) {
    auto it = this->seqs.find(seq);
    TORCH_CHECK(it != this->seqs.end(), "FileBucket: seq ", seq, " not begun");
    return *it->second;
}

const BucketFile& FileBucket::file(int64_t seq) const {
    auto it = this->seqs.find(seq);
    TORCH_CHECK(it != this->seqs.end(), "FileBucket: seq ", seq, " not begun");
    return *it->second;
}

void FileBucket::begin(int64_t seq) {
    if (!this->enabled()) {
        return;
    }
    TORCH_CHECK(this->seqs.find(seq) == this->seqs.end(), "FileBucket: seq ", seq, " already begun");
    auto f = std::make_unique<BucketFile>();
    f->path = (std::filesystem::path(this->spill_dir) / std::to_string(seq)).string();
    f->fd = ::open(f->path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    TORCH_CHECK(f->fd >= 0, "FileBucket: cannot open ", f->path);
    this->seqs.emplace(seq, std::move(f));
}

void FileBucket::end(int64_t seq) {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return;
    }
    BucketFile& f = *it->second;
    if (f.fd >= 0) {
        ::close(f.fd);
    }
    std::error_code ec;
    std::filesystem::remove(f.path, ec);
    this->seqs.erase(it);
}

int64_t
FileBucket::claim(BucketFile& f, int64_t page_id, std::vector<int64_t> token_ids, double mass, int64_t mass_step) {
    TORCH_CHECK(f.find(page_id) < 0, "FileBucket: page ", page_id, " already held");
    const int64_t cnt = static_cast<int64_t>(token_ids.size());
    const int64_t nbytes = this->page_nbytes(cnt);
    const int64_t off = f.write_offset;
    f.write_offset += nbytes;
    BucketPage pg{page_id, std::move(token_ids), off, nbytes, mass, mass_step};
    auto it = std::lower_bound(f.pages.begin(), f.pages.end(), page_id, [](const BucketPage& p, int64_t idx) {
        return p.page_id < idx;
    });
    f.pages.insert(it, std::move(pg));
    return off;
}

void FileBucket::write_page(BucketFile& f, int64_t offset, const at::Tensor& k, const at::Tensor& v) {
    int64_t at_off = offset;
    auto write_tensor = [&](const at::Tensor& t) {
        at::Tensor c = t.to(at::kCPU, this->dtype).contiguous();
        const int64_t nb = c.numel() * at::elementSize(this->dtype);
        const ssize_t w = ::pwrite(f.fd, c.data_ptr(), nb, at_off);
        TORCH_CHECK(w == nb, "FileBucket: short write (", w, " of ", nb, ")");
        at_off += nb;
    };
    write_tensor(k);
    write_tensor(v);
}

BucketView FileBucket::view(int64_t seq) const {
    BucketView v;
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return v;
    }
    const BucketFile& f = *it->second;
    v.pages.reserve(f.pages.size());
    for (const BucketPage& p : f.pages) {
        v.pages.push_back(BucketPageRef{p.page_id, &p.token_ids});
    }
    return v;
}

Reservation FileBucket::reserve(int64_t seq, const DemotedKV& demoted, int64_t step) {
    Reservation r;
    if (demoted.num_tokens() == 0) {
        return r;
    }
    TORCH_CHECK(this->enabled(), "FileBucket: reserve on a disabled bucket");
    BucketFile& f = this->file(seq);
    std::vector<DemotedPage> pages = demoted_pages(demoted, this->page_size, "FileBucket::reserve");
    const size_t n_pages = pages.size();
    r.page_ids.resize(n_pages);
    r.src_bases.resize(n_pages);
    r.dst_bases.resize(n_pages);
    for (size_t pi = 0; pi < n_pages; ++pi) {
        DemotedPage& src = pages[pi];
        r.page_ids[pi] = src.page_id;
        r.src_bases[pi] = src.src_base;
        r.dst_bases[pi] = this->claim(f, src.page_id, std::move(src.token_ids), src.mass, step);
    }
    return r;
}

void FileBucket::fill(int64_t seq, const ActiveBuffer& active, const Reservation& r) {
    if (r.page_ids.empty()) {
        return;
    }
    BucketFile& f = this->file(seq);
    const int64_t ps = this->page_size;
    at::Tensor k_src = active.k_pool_all().view({this->n_layers, -1, this->n_kv_heads, this->head_dim});
    at::Tensor v_src = active.v_pool_all().view({this->n_layers, -1, this->n_kv_heads, this->head_dim});
    for (size_t pi = 0; pi < r.page_ids.size(); ++pi) {
        // The active pool is layer-major, which is the on-disk layout, so the write
        // takes the slot block of every layer as it stands.
        this->write_page(
            f,
            r.dst_bases[pi],
            k_src.narrow(1, r.src_bases[pi], ps),
            v_src.narrow(1, r.src_bases[pi], ps)
        );
    }
}

void FileBucket::accept(int64_t seq, const PageKV& page) {
    TORCH_CHECK(this->enabled(), "FileBucket: accept on a disabled bucket");
    BucketFile& f = this->file(seq);
    at::Tensor token_ids_c = page.token_ids.to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> token_ids(
        token_ids_c.data_ptr<int64_t>(),
        token_ids_c.data_ptr<int64_t>() + token_ids_c.numel()
    );
    const int64_t off = this->claim(f, page.page_id, token_ids, page.mass, page.mass_step);
    // page.k/v are page-major [cnt, n_layers, ...]; the file is layer-major.
    this->write_page(f, off, page.k.transpose(0, 1), page.v.transpose(0, 1));
}

void FileBucket::read_kv(int64_t seq, int64_t page_id, at::Tensor& k_out, at::Tensor& v_out) const {
    const BucketFile& f = this->file(seq);
    const int64_t idx = f.find(page_id);
    TORCH_CHECK(idx >= 0, "FileBucket: page ", page_id, " not held");
    const BucketPage& pg = f.pages[idx];
    const int64_t cnt = static_cast<int64_t>(pg.token_ids.size());
    // The pread destination, in the on-disk shape: layer-major K then layer-major V.
    // A page carrier is page-major, so each half transposes out of it; the halves keep
    // raw alive, which is what the transpose may alias.
    at::Tensor raw = at::empty(
        {2, this->n_layers, cnt, this->n_kv_heads, this->head_dim},
        at::TensorOptions().dtype(this->dtype).device(at::kCPU)
    );
    const ssize_t r = ::pread(f.fd, raw.data_ptr(), pg.nbytes, pg.offset);
    TORCH_CHECK(r == pg.nbytes, "FileBucket: short read (", r, " of ", pg.nbytes, ") for page ", page_id);
    k_out = raw.select(0, 0).transpose(0, 1).contiguous();
    v_out = raw.select(0, 1).transpose(0, 1).contiguous();
}

GatheredK
FileBucket::gather_k(int64_t seq, const std::vector<int64_t>& page_ids, const std::vector<int64_t>& layers) const {
    const int64_t ps = this->page_size;
    const int64_t n = static_cast<int64_t>(page_ids.size());
    auto opts = at::TensorOptions().dtype(this->dtype).device(at::kCPU);
    GatheredK out;
    // A pread lands wherever it is told, so a file bucket reads straight into the
    // requested order.
    out.order.resize(page_ids.size());
    std::iota(out.order.begin(), out.order.end(), 0);
    out.layers.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        out.layers.push_back(at::empty({n, ps, this->n_kv_heads, this->head_dim}, opts));
    }
    if (n == 0) {
        return out;
    }
    const BucketFile& f = this->file(seq);
    const int64_t elt = at::elementSize(this->dtype);
    // The API is batched; a file bucket loops internally, reading only the requested
    // layers' K blocks straight into their destination rows (on-disk layout == pool
    // layout, so nothing is restaged).
    for (int64_t j = 0; j < n; ++j) {
        const int64_t idx = f.find(page_ids[j]);
        TORCH_CHECK(idx >= 0, "FileBucket: page ", page_ids[j], " not held");
        const BucketPage& pg = f.pages[idx];
        const int64_t cnt = static_cast<int64_t>(pg.token_ids.size());
        TORCH_CHECK(
            cnt == ps,
            "FileBucket::gather_k: page ",
            pg.page_id,
            " holds ",
            cnt,
            " tokens, expected a full page of ",
            ps
        );
        const int64_t nb = this->n_kv_heads * cnt * this->head_dim * elt;
        for (size_t li = 0; li < layers.size(); ++li) {
            at::Tensor row = out.layers[li].select(0, j);
            const ssize_t r = ::pread(f.fd, row.data_ptr(), nb, pg.offset + layers[li] * nb);
            TORCH_CHECK(
                r == nb,
                "FileBucket: short read (",
                r,
                " of ",
                nb,
                ") for page ",
                pg.page_id,
                " layer ",
                layers[li]
            );
        }
    }
    return out;
}

std::vector<PageKV> FileBucket::take_pages(int64_t seq, const std::vector<int64_t>& page_ids) {
    if (page_ids.empty()) {
        return {};
    }
    BucketFile& f = this->file(seq);
    std::vector<PageKV> out;
    out.reserve(page_ids.size());
    for (int64_t pid : page_ids) {
        const int64_t idx = f.find(pid);
        TORCH_CHECK(idx >= 0, "FileBucket: page ", pid, " not held");
        PageKV pg;
        pg.page_id = pid;
        pg.token_ids = at::tensor(f.pages[idx].token_ids, at::TensorOptions().dtype(at::kLong));
        pg.mass = f.pages[idx].mass;
        pg.mass_step = f.pages[idx].mass_step;
        this->read_kv(seq, pid, pg.k, pg.v);
        f.pages.erase(f.pages.begin() + idx);
        out.push_back(std::move(pg));
    }
    return out;
}

std::vector<PageKV> FileBucket::spill_to_budget(int64_t seq, int64_t budget_tokens, int64_t step, double decay) {
    std::vector<PageKV> out;
    if (budget_tokens <= 0 || decay >= 1.0) {
        return out;
    }
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return out;
    }
    BucketFile& f = *it->second;
    int64_t tokens = this->bucket_tokens(seq);
    if (tokens <= budget_tokens) {
        return out;
    }
    // f.pages is already page-id sorted and the sort is stable, so eff-mass ties
    // break by ascending page id.
    std::vector<PageMass> mass;
    mass.reserve(f.pages.size());
    for (const BucketPage& pg : f.pages) {
        mass.push_back(PageMass{pg.mass, pg.mass_step});
    }
    const std::vector<int64_t> order = coldest_first(mass, step, decay);
    std::vector<int64_t> victims;
    for (size_t oi = 0; oi < order.size() && tokens > budget_tokens; ++oi) {
        const BucketPage& pg = f.pages[order[oi]];
        tokens -= static_cast<int64_t>(pg.token_ids.size());
        victims.push_back(pg.page_id);
    }
    return this->take_pages(seq, victims);
}

void FileBucket::release_pending(int64_t seq) {
    (void)seq;
}

bool FileBucket::has_seq(int64_t seq) const {
    return this->seqs.find(seq) != this->seqs.end();
}

bool FileBucket::has_page(int64_t seq, int64_t page_id) const {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return false;
    }
    return it->second->find(page_id) >= 0;
}

int64_t FileBucket::bucket_tokens(int64_t seq) const {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return 0;
    }
    int64_t n = 0;
    for (const BucketPage& p : it->second->pages) {
        n += static_cast<int64_t>(p.token_ids.size());
    }
    return n;
}

int64_t FileBucket::size() const {
    int64_t n = 0;
    for (const auto& [seq, f] : this->seqs) {
        for (const BucketPage& p : f->pages) {
            n += static_cast<int64_t>(p.token_ids.size());
        }
    }
    return n;
}

}  // namespace pulsar
