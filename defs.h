// defs.h
#pragma once
#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;

#define MAX_GPU_CNT         32
#define STEP_CNT            1000
#define JMP_CNT             512
#define MD_LEN              12

// Tesla T4 = SM 7.5 → enable OLD_GPU
#define OLD_GPU

#ifdef __CUDA_ARCH__
    #ifdef OLD_GPU
        #define BLOCK_SIZE      256
        #define PNT_GROUP_CNT   4   // Critical for T4 register pressure
    #else
        #define BLOCK_SIZE      256
        #define PNT_GROUP_CNT   24
    #endif
#else
    #define BLOCK_SIZE          512
    #define PNT_GROUP_CNT       64
#endif

#define GPU_DP_SIZE         48
#define MAX_DP_CNT          (256 * 1024)
#define JMP_MASK            (JMP_CNT - 1)
#define DPTABLE_MAX_CNT     16
#define MAX_CNT_LIST        (512 * 1024)
#define DP_FLAG             0x8000
#define INV_FLAG            0x4000
#define JMP2_FLAG           0x2000

#define TAME                0
#define WILD1               1
#define WILD2               2

struct TKparams {
    u64* Kangs;
    u32 KangCnt;
    u32 BlockCnt;
    u32 BlockSize;
    u32 GroupCnt;
    u64* L2;
    u64 DP;
    u32* DPs_out;
    u64* Jumps1;
    u64* Jumps2;
    u64* Jumps3;
    u64* JumpsList;
    u32* DPTable;
    u32* L1S2;
    u64* LastPnts;
    u64* LoopTable;
    u32* dbg_buf;
    u32* LoopedKangs;
    bool IsGenMode;
    u32 KernelA_LDS_Size;
    u32 KernelB_LDS_Size;
    u32 KernelC_LDS_Size;
};