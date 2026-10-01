#pragma once

#include <concepts>
#include <cstdint>
#include <utility>
#include <vector>

// The address-cursor read layer: a std::map-shaped view over a component's pages
// keyed by page index. A token's ADDRESS is its index in the whole conversation;
// page index = address / page_size is a page's permanent identity. Every component
// that holds pages exposes one of these, so KVMemory::history merge-walks them
// uniformly and never touches K/V or disk.

namespace pulsar {

// A page as seen through a cursor: a non-owning handle to the page's token ids.
// The backing storage (the ActiveBuffer's tok[] / a bucket's RAM index) outlives
// the walk.
struct PageRef {
    const std::vector<int64_t>* ids;
    const std::vector<int64_t>& token_ids() const {
        return *this->ids;
    }
};

// *cursor value: .first = page index, .second = page (mirrors std::map value_type).
using PageEntry = std::pair<const int64_t, PageRef>;

// Mirrors std::map's const_iterator over pages: dereference to {index, page},
// pre-increment to the next page, compare against end().
template <typename It>
concept PageCursor = requires(It it) {
    { *it } -> std::convertible_to<PageEntry>;
    { ++it } -> std::same_as<It&>;
    { it == it } -> std::same_as<bool>;
};

// Page cursors keyed by page index (mirror std::map). lower_bound(idx) is the first
// page with index >= idx; half-open ranges are
// [lower_bound(start), lower_bound(end)).
template <typename C>
concept PageIndexed = requires(const C& c, int64_t idx) {
    { c.lower_bound(idx) } -> PageCursor;
    { c.end() } -> PageCursor;
};

// A page as a bucket read cursor sees it: page index + a non-owning pointer to its
// RAM token ids. The bucket's KV (host pool or disk) is NOT touched by the cursor.
struct BucketPageRef {
    int64_t page_id;
    const std::vector<int64_t>* token_ids;
};

// A cursor over one sequence's bucket pages, ascending by page index. Owns the
// BucketPageRef vector; the token ids it points at stay in the bucket's RAM index.
struct BucketView {
    std::vector<BucketPageRef> pages;  // ascending by page index

    struct Cursor {
        const std::vector<BucketPageRef>* pages;
        int64_t i;

        PageEntry operator*() const {
            return PageEntry((*this->pages)[this->i].page_id, PageRef{(*this->pages)[this->i].token_ids});
        }
        Cursor& operator++() {
            ++this->i;
            return *this;
        }
        bool operator==(const Cursor& o) const {
            return this->i == o.i && this->pages == o.pages;
        }
    };

    Cursor lower_bound(int64_t idx) const {
        int64_t lo = 0, hi = static_cast<int64_t>(this->pages.size());
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (this->pages[mid].page_id < idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return Cursor{&this->pages, lo};
    }
    Cursor end() const {
        return Cursor{&this->pages, static_cast<int64_t>(this->pages.size())};
    }
};
static_assert(PageIndexed<BucketView>);

}  // namespace pulsar
