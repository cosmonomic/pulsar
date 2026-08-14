#include "host_pages.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>

namespace pulsar {

std::vector<DemotedPage> demoted_pages(const DemotedKV& demoted, int64_t page_size, const char* who) {
    std::vector<DemotedPage> out;
    const int64_t M = demoted.num_tokens();
    if (M == 0) {
        return out;
    }
    const int64_t ps = page_size;
    TORCH_CHECK(M % ps == 0, who, ": demoted token count ", M, " must be a multiple of page_size ", ps);
    const int64_t n_pages = M / ps;
    at::Tensor addr_c = demoted.addr().to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> addr(addr_c.data_ptr<int64_t>(), addr_c.data_ptr<int64_t>() + addr_c.numel());
    at::Tensor tok_c = demoted.tok().to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> tok(tok_c.data_ptr<int64_t>(), tok_c.data_ptr<int64_t>() + tok_c.numel());
    at::Tensor slots_c = demoted.victim_slots.to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> slots(slots_c.data_ptr<int64_t>(), slots_c.data_ptr<int64_t>() + slots_c.numel());
    at::Tensor mass_raw = demoted.page_mass();
    std::vector<float> mass;
    if (mass_raw.defined() && mass_raw.numel() == n_pages) {
        at::Tensor mt = mass_raw.to(at::kCPU, at::kFloat).contiguous();
        mass.assign(mt.data_ptr<float>(), mt.data_ptr<float>() + n_pages);
    }
    out.resize(n_pages);
    for (int64_t pi = 0; pi < n_pages; ++pi) {
        const int64_t lo = pi * ps;
        const int64_t base = addr[lo];
        TORCH_CHECK(
            base % ps == 0 && addr[lo + ps - 1] == base + ps - 1,
            who,
            ": demoted page not address-aligned (base ",
            base,
            ")"
        );
        const int64_t sbase = slots[lo];
        TORCH_CHECK(
            slots[lo + ps - 1] == sbase + ps - 1,
            who,
            ": victim page slots not a contiguous ps-run (base ",
            sbase,
            ")"
        );
        DemotedPage& d = out[pi];
        d.page_id = base / ps;
        d.token_ids.assign(tok.begin() + lo, tok.begin() + lo + ps);
        d.mass = pi < static_cast<int64_t>(mass.size()) ? static_cast<double>(mass[pi]) : 0.0;
        d.src_base = sbase;
    }
    return out;
}

std::vector<int64_t> coldest_first(const std::vector<PageMass>& pages, int64_t step, double decay) {
    const int64_t np = static_cast<int64_t>(pages.size());
    std::vector<double> eff(np);
    for (int64_t i = 0; i < np; ++i) {
        const int64_t age = std::max<int64_t>(0, step - pages[i].mass_step);
        eff[i] = pages[i].mass * std::pow(1.0 - decay, static_cast<double>(age));
    }
    std::vector<int64_t> order(np);
    for (int64_t i = 0; i < np; ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return eff[a] < eff[b]; });
    return order;
}

}  // namespace pulsar
