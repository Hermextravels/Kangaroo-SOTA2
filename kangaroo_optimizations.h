# pragma once

// Strategy 1: Maximize Computational Throughput
#define USE_GLV_ENDOMORPHISM 1
#define USE_NEGATION_MAP 1
#define USE_BATCH_INVERSION 1

// GLV endomorphism constants for secp256k1
#define GLV_BETA "7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee"
#define GLV_LAMBDA1 "5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72"
#define GLV_LAMBDA2 "ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce"

// Strategy 2: Four-Kangaroo Method Configuration
#define KANGAROO_COUNT 4
#define OPTIMAL_MEAN_DISTANCE 0.75  // Optimal ratio found through experimentation

// Strategy 3: Enhanced Distributed Infrastructure
#define DP_BITS 16                  // Configurable distinguished point bits
#define BATCH_SIZE 2048            // Number of points to process in parallel
#define THREAD_BLOCK_SIZE 256      // Optimal for modern NVIDIA GPUs
#define USE_SLOPPY_REDUCTION 1     // Use faster but slightly less accurate reduction
#define COLLISION_VERIFY_BITS 32   // Number of bits to use for quick collision check

// Strategy 4: Time-Memory Trade-off Parameters
#define TMTO_ENABLE 1              // Enable time-memory trade-off
#define TMTO_TABLE_BITS 24         // Size of pre-computed table (2^24 entries)
#define TMTO_CHAINS 16             // Number of chains per start point

// Performance Optimization Flags
#define USE_FAST_REDUCTION 1       // Use faster modular reduction
#define USE_COMPRESSED_POINTS 1    // Use point compression when possible
#define USE_PARALLEL_FORMULAS 1    // Use parallel addition formulas
#define MAX_REDUCTIONS_BEFORE_MOD 4 // Maximum accumulated values before modulo

// Memory Management
#define COALESCED_MEMORY_ACCESS 1  // Ensure coalesced memory access patterns
#define USE_SHARED_MEMORY_CACHE 1  // Cache frequently used data in shared memory
#define POINTS_PER_THREAD 8        // Number of points handled per thread

// Advanced Optimizations
#define USE_ENDOMORPHISM_PRECOMP 1 // Precompute endomorphism values
#define USE_KARATSUBA_MULTIPLY 1   // Use Karatsuba multiplication for large integers
#define PARALLEL_COLLISION_CHECK 1  // Check collisions in parallel