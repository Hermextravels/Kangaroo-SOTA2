// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


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

// Forward declaration
class EcInt;

// Checkpoint structure for resumable operations
struct CheckpointData {
    u64 totalOps;
    u32 pntIndex;
    u8 checkpointVersion;
    u64 currentStartWords[4];  // 256-bit number stored as 4 u64
    u64 currentEndWords[4];    // 256-bit number stored as 4 u64
};


#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
#define STEP_CNT			1000

#define JMP_CNT				512  // Optimized for T4 constant memory

// Memory management for T4
#define JMP_TABLE_SPLIT    1
#define JMP_BATCH_SIZE     256
#define USE_SHARED_MEM_CACHE 1

//use different options for cards older than RTX 40xx
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ == 750  // Tesla T4
        #define BLOCK_SIZE           256
        #define PNT_GROUP_CNT       32
    #elif __CUDA_ARCH__ < 890
        #define OLD_GPU
        #define BLOCK_SIZE          512
        #define PNT_GROUP_CNT       64
    #else
        #define BLOCK_SIZE          256
        #define PNT_GROUP_CNT       24
    #endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif


// Kangaroo types for four-kangaroo method
#define TAME				0  // Tame kang 1
#define TAME2				1  // Tame kang 2
#define WILD1				2  // Wild kang 1
#define WILD2				3  // Wild kang 2

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000

#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;
	u64* L2;
	u64 DP;
	u32* DPs_out;
	u64* Jumps1; //x(32b), y(32b), d(32b)
	u64* Jumps2; //x(32b), y(32b), d(32b)
	u64* Jumps3; //x(32b), y(32b), d(32b)
	u64* JumpsList; //list of all performed jumps, grouped by warp(32) every 8 groups (from PNT_GROUP_CNT). Each jump is 2 bytes: 10bit jump index + flags: INV_FLAG, DP_FLAG, JMP2_FLAG
	u32* DPTable;
	u32* L1S2;
	u64* LastPnts;
	u64* LoopTable;
	u32* dbg_buf;
	u32* LoopedKangs;
	bool IsGenMode; //tames generation mode

	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;	
};

