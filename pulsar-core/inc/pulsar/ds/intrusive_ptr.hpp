#pragma once

#include <atomic>
#include <concepts>
#include <type_traits>
#include <utility>

namespace pulsar::ds {

template <typename type> constexpr bool atomic_uint = false;

template <std::unsigned_integral type>
    requires(!std::same_as<type, bool>)
constexpr bool atomic_uint<std::atomic<type>> = true;

template <typename type>
concept refcount_target = atomic_uint<decltype(type::ref)> &&
    (std::is_final_v<type> || std::has_virtual_destructor_v<type>);

template <typename target_type> class intrusive_ptr {
    target_type* p = nullptr;

  public:
    using element_type = target_type;

    intrusive_ptr() noexcept = default;

    explicit intrusive_ptr(target_type* value) noexcept
    : p(value) {
        static_assert(refcount_target<target_type>);
        if (p) {
            p->ref.fetch_add(1, std::memory_order_relaxed);
        }
    }

    intrusive_ptr(const intrusive_ptr& other) noexcept
    : intrusive_ptr(other.p) {}

    intrusive_ptr(intrusive_ptr&& other) noexcept
    : p(std::exchange(other.p, nullptr)) {}

    ~intrusive_ptr() noexcept {
        static_assert(refcount_target<target_type>);
        if (p && p->ref.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete p;
        }
    }

    friend void swap(intrusive_ptr& a, intrusive_ptr& b) noexcept {
        std::swap(a.p, b.p);
    }

    intrusive_ptr& operator=(intrusive_ptr other) noexcept {
        std::ranges::swap(*this, other);
        return *this;
    }

    target_type* get() const noexcept {
        return p;
    }

    target_type* operator->() const noexcept {
        return p;
    }

    target_type& operator*() const noexcept {
        return *p;
    }

    explicit operator bool() const noexcept {
        return p != nullptr;
    }

    friend bool operator==(const intrusive_ptr& a, const intrusive_ptr& b) noexcept {
        return a.p == b.p;
    }
};

}  // namespace pulsar::ds
