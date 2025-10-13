// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

// Include CUDA headers first
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#endif

// Default configurations
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef JMP_CNT
#define JMP_CNT 256
#endif

#ifndef PNT_GROUP_CNT
#define PNT_GROUP_CNT 4
#endif

#ifndef INV_FLAG
#define INV_FLAG 0x8000
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;
#else
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;
#endif

#ifdef __cplusplus
}
#endif

// Project-wide defines
// Only define these once, and let CUDA arch-specific code override as needed
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef JMP_CNT
#define JMP_CNT 256
#endif
#ifndef PNT_GROUP_CNT
#define PNT_GROUP_CNT 4
#endif
#ifndef INV_FLAG
#define INV_FLAG 0x8000
#endif
#ifndef JMP3_FLAG
#define JMP3_FLAG 0x1000
#endif
#ifndef JMP4_FLAG
#define JMP4_FLAG 0x0800
#endif
#ifndef JMP4_TRIGGER_MASK
#define JMP4_TRIGGER_MASK 0x0F00
#endif
#ifndef EMERGENCY_ESCAPE
#define EMERGENCY_ESCAPE 4
#endif

#ifndef __CUDACC__
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;
#else
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;
#endif



#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
#define STEP_CNT			1000

// #define JMP_CNT				512
#define JMP4_CNT            512  // Size of emergency escape jump table

//use different options for cards older than RTX 40xx
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		// Tesla T4 optimized settings (Compute 7.5, 40 SMs)
		#if __CUDA_ARCH__ == 750
			#define BLOCK_SIZE			512
			// Optimized for T4: 40 SMs, 512 threads = 20480 total threads
			// 64 groups gives best occupancy on T4
			#define PNT_GROUP_CNT		64
		#else
			#define BLOCK_SIZE			512
			//can be 8, 16, 24, 32, 40, 48, 56, 64
			#define PNT_GROUP_CNT		64
		#endif
	#else
		#define BLOCK_SIZE			256
		//can be 8, 16, 24, 32
		#define PNT_GROUP_CNT		24
	#endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2
#define WILD3               3  // Wild kangs3 (New: Fourth kangaroo type)

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG            0x2000
#define JMP3_FLAG            0x1000 // New
#define JMP4_FLAG            0x0800 // New


#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
// In defs.h, add Jumps4 to TKparams struct
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
    u64* Jumps4; //emergency escape jumps table
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