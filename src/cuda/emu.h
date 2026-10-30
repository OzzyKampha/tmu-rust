// CPU shim that lets kernels.cu compile as plain C++ (`--features cuda-emu`).
//
// Execution model: blocks run one after another; the `blockDim.x` threads of a
// block run concurrently on OS threads and meet at `__syncthreads()` through a
// std::barrier. Because only one block is ever live, `__shared__` variables can
// simply be function-local statics (shared by the block's threads), and dynamic
// shared memory is one host buffer per launch.
#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

struct tmu_dim3 {
    unsigned int x, y, z;
};

extern thread_local tmu_dim3 threadIdx;
extern tmu_dim3 blockIdx;
extern tmu_dim3 blockDim;
extern tmu_dim3 gridDim;
extern unsigned long long* tmu_dyn_smem;
extern std::barrier<>* tmu_block_barrier;

#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__ __restrict

inline void __syncthreads() {
    if (blockDim.x > 1) tmu_block_barrier->arrive_and_wait();
}

inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
inline int atomicMin(int* p, int v) {
    int cur = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v < cur && !__atomic_compare_exchange_n(p, &cur, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    return cur;
}
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }

// ---- launcher ---------------------------------------------------------------

typedef void (*tmu_emu_fn)(const void*);
struct TmuEmuEntry {
    const char* name;
    tmu_emu_fn fn;
    size_t param_size;
};
