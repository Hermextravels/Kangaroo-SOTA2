// CPU-only stubs to satisfy linker when building without CUDA
#include "GpuKang.h"
#include "Ec.h"
#ifdef NO_CUDA
#include "cuda_stub.h"

// Stub implementations for GPU functions. They do nothing but allow linking.
cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
    return cudaSuccess;
}

void CallGpuKernelGen(TKparams Kparams)
{
    // No-op for CPU build
}

void CallGpuKernelABC(TKparams Kparams)
{
}

#endif // NO_CUDA
