#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// Shared m16n8k16 tensor-core MMA device helpers for the sm_120 (consumer
// Blackwell, Ampere-class MMA) kernels: bf16/fp16 operands, fp32 accumulate.

namespace pulsar {
namespace mma {

__device__ __forceinline__ uint32_t cvta(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Load a 16x16 b16 shared tile as four 8x8 quadrants, ordered
// (rows0-7,cols0-7),(rows8-15,cols0-7),(rows0-7,cols8-15),(rows8-15,cols8-15).
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* p) {
    uint32_t a = cvta(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const void* p) {
    uint32_t a = cvta(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// D[16x8] = A[16x16] * B[8x16]^col + C[16x8], fp32 accumulate. BF16 selects the
// bf16 operand form; false selects fp16. A is 4 regs (.row), B is 2 regs (.col).
template <bool BF16>
__device__ __forceinline__ void mma_m16n8k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]);
template <>
__device__ __forceinline__ void
mma_m16n8k16<true>(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
template <>
__device__ __forceinline__ void
mma_m16n8k16<false>(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

}  // namespace mma
}  // namespace pulsar
