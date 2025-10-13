#pragma once

// T4-specific memory management
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 750

// Shared memory cache for jump tables
__device__ __shared__ u64 shared_jmp_cache[JMP_BATCH_SIZE * 4];

// Load jump table data into shared memory
__device__ inline void load_jump_table_batch(int batch_idx) {
    int tid = threadIdx.x;
    if (tid < JMP_BATCH_SIZE) {
        int idx = batch_idx * JMP_BATCH_SIZE + tid;
        if (idx < JMP_CNT) {
            shared_jmp_cache[tid * 4 + 0] = jmp2_table_part1[idx * 2 + 0];
            shared_jmp_cache[tid * 4 + 1] = jmp2_table_part1[idx * 2 + 1];
            shared_jmp_cache[tid * 4 + 2] = jmp2_table_part2[idx * 2 + 0];
            shared_jmp_cache[tid * 4 + 3] = jmp2_table_part2[idx * 2 + 1];
        }
    }
    __syncthreads();
}

// Get jump table entry from shared memory
__device__ inline void get_jump_entry(int idx, u64* out) {
    int cache_idx = idx % JMP_BATCH_SIZE;
    out[0] = shared_jmp_cache[cache_idx * 4 + 0];
    out[1] = shared_jmp_cache[cache_idx * 4 + 1];
    out[2] = shared_jmp_cache[cache_idx * 4 + 2];
    out[3] = shared_jmp_cache[cache_idx * 4 + 3];
}

#endif
#endif