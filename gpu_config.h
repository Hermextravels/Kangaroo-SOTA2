# pragma once

// Tesla T4 Configuration (Compute Capability 7.5)
#define T4_WARP_SIZE 32
#define T4_MAX_THREADS_PER_BLOCK 1024
#define T4_MAX_BLOCKS_PER_SM 16
#define T4_MAX_SHARED_MEMORY_PER_BLOCK 49152
#define T4_L1_CACHE_SIZE 32768
#define T4_MAX_REGISTERS_PER_BLOCK 65536

// Optimized configurations for T4
#define USE_TENSOR_CORES 0  // T4 has Tensor Cores but we don't use them for this workload
#define USE_MIXED_PRECISION 0
#define USE_SHUFFLE_SYNC 1  // Use efficient warp shuffle operations
#define USE_COOPERATIVE_GROUPS 1

// Memory access patterns
#define COALESCE_MEMORY_ACCESS 1
#define USE_SHARED_MEMORY_CACHE 1
#define L1_CACHE_PREFERENCE cudaFuncCachePreferL1

// Thread configuration
#ifndef OLD_GPU
    #define BLOCK_SIZE 256            // Optimal for T4
    #define PNT_GROUP_CNT 32         // Adjusted for T4's memory hierarchy
    #define WARPS_PER_BLOCK 8        // 256/32
    #define MAX_BLOCKS_PER_SM 16     // T4 specific
#endif

// Register optimization
#define MAX_REGISTERS_PER_THREAD 64  // Limit register usage for better occupancy