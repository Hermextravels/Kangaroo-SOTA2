// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "defs.h"
#include "RCGpuUtils.h"

// JMP3_FLAG (0x1000) and JMP4_FLAG (0x0800) are now correctly included from defs.h

//imp2 table points for KernelA
__device__ __constant__ u64 jmp2_table[8 * JMP_CNT];


#define BLOCK_CNT	gridDim.x
#define BLOCK_X		blockIdx.x
#define THREAD_X	threadIdx.x

//coalescing
#define LOAD_VAL_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); }
#define SAVE_VAL_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); }


extern __shared__ u64 LDS[]; 

// Placeholder helper functions from RCGpuUtils.h/RCGpuCore.cu context (required for compilation)
extern "C" __device__ void TPointPriv_Add_cuda(u64* x, u64* y, u64* priv, u64* x_add, u64* y_add, u64* inverse);
extern "C" __device__ void MulPnt_cuda(u64* x, u64* y, u64* x_mul, u64* y_mul, u64* inverse);
extern "C" __device__ void Copy_int4_x2(u64* dst, const u64* src);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU

//this kernel performs main jumps (New GPU optimization path)
// T4 optimization: __launch_bounds__(512, 2) allows 2 blocks per SM for better occupancy
extern "C" __global__ __launch_bounds__(512, 2) void KernelA(const TKparams Kparams)
{
	u64* LDS_x = (u64*)LDS;
	u64* LDS_y = (u64*)(LDS + 4 * JMP_CNT);
	u64* LDS_inv = (u64*)(LDS_y + 4 * JMP_CNT);
	u64* jmp1_table = LDS_x;

	int global_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	u64* x_kangs = Kparams.Kangs + global_ind * 8; //x
	u64* y_kangs = x_kangs + 4; //y
	u64* priv_kangs = x_kangs + 8; //priv

	u64* L2x = (u64*)Kparams.L2;
	u64* L2y = L2x + Kparams.KangCnt * 4;
	u64* L2inv = L2y + Kparams.KangCnt * 4;

	u64* L2x_cur = L2x + global_ind * 4;
	u64* L2y_cur = L2y + global_ind * 4;
	u64* L2inv_cur = L2inv + global_ind * 4;

	u32* jumps_list = (u32*)Kparams.JumpsList + global_ind * STEP_CNT;

	//load jumps1 to LDS
	for (int i = THREAD_X; i < JMP_CNT * 4; i += BLOCK_SIZE)
	{
		jmp1_table[i] = ((u64*)Kparams.Jumps1)[i];
		LDS_y[i] = ((u64*)Kparams.Jumps1)[i + JMP_CNT * 4];
	}
	__syncthreads();

	//load inv from L2 cache (last step)
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256(LDS_inv + group * 4, L2inv, group);
	}
	__syncthreads();

	u64 x[4], y[4], priv[4];
	u64 x0[4], y0[4];

	u32 jmp_ind;
    u32 thread_base_kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);

	//load Kangs from global
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256(x, x_kangs + group * 4, 0);
		LOAD_VAL_256(y, y_kangs + group * 4, 0);
		LOAD_VAL_256(priv, priv_kangs + group * 4, 0);
		
		Copy_int4_x2(L2x_cur + group * 4, x); //store x to L2
		Copy_int4_x2(L2y_cur + group * 4, y); //store y to L2

		TPointPriv_Add_cuda(x, y, priv, x0, y0, LDS_inv + group * 4); //calc next point

		SAVE_VAL_256(x_kangs + group * 4, x, 0); //save next x to global
		SAVE_VAL_256(y_kangs + group * 4, y, 0); //save next y to global
		
		Copy_int4_x2(LDS_inv + group * 4, x0); //store x0 to LDS
	}
	__syncthreads();

	u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];

    // --- JUMP TABLE SELECTION FOR 4 KANGAROO GROUPS ---
    // The kangs are partitioned into 4 quarters: TAME (0), WILD1 (1), WILD2 (2), WILD3 (3)
    const u32 GRP_QUARTER_SIZE = Kparams.KangCnt / 4; 
    u32 thread_group_quarter = thread_base_kang_ind / GRP_QUARTER_SIZE;

    u64* Jumps_P; // Primary Jump Table (J1, J2, J3, J4)
    u64* Jumps_A; // Alternative Jump Table (J2 for TAME, J1 for WILDs)
    
    // Determine Jumps_P and Jumps_A pointers based on kangaroo type (0, 1, 2, or 3)
    if (thread_group_quarter == TAME) { // 0: Uses J1 (Jumps1) primarily
        Jumps_P = jmp1_table; 
        Jumps_A = jmp2_table; // TAME uses J2 as alternative (J2 flag applies)
    } else if (thread_group_quarter == WILD1) { // 1: Uses J2 (jmp2_table) primarily
        Jumps_P = jmp2_table;
        Jumps_A = jmp1_table; // WILDs use J1 as alternative (J2 flag applies)
    } else if (thread_group_quarter == WILD2) { // 2: Uses J3 (Kparams.Jumps3) primarily
        Jumps_P = (u64*)Kparams.Jumps3; 
        Jumps_A = jmp1_table;
    } else { // WILD3 (3): Uses J4 (Kparams.Jumps4) primarily
        Jumps_P = (u64*)Kparams.Jumps4;
        Jumps_A = jmp1_table;
    }
    // -----------------------------------------------------

	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
	{
		__align__(16) u64 inverse[5];
		u64* jmp_table;
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		
		//first group
		LOAD_VAL_256(x, L2x_cur, 0);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> 0) & 1) ? Jumps_A : Jumps_P; // Use determined tables
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
		TPointPriv_Add_cuda(x, y, priv, jmp_x, jmp_y, inverse);

		//other groups
		for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
		{
			//load current point
			LOAD_VAL_256(x0, L2x_cur, group);
			LOAD_VAL_256(y0, L2y_cur, group);
			
			//save point for next step (x, y)
			SAVE_VAL_256(x_kangs + group * 4, x, 0); 
			SAVE_VAL_256(y_kangs + group * 4, y, 0);

			//calc next point
			TPointPriv_Add_cuda(x, y, priv, x0, y0, inverse);

			//save x0 for next step to LDS (for inv calc)
			Copy_int4_x2(LDS_inv + group * 4, x0);
			
			//calc jmp index
			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? Jumps_A : Jumps_P; // Use determined tables
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			
			u32 jmp_list_ind = jmp_ind;

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1u << group);
				// --- Set correct flag based on Primary Jump Table used when loop broke ---
                if (thread_group_quarter == TAME || thread_group_quarter == WILD1) {
                    jmp_ind |= JMP2_FLAG;
                } else if (thread_group_quarter == WILD2) {
                    jmp_ind |= JMP3_FLAG; 
                } else { // WILD3
                    jmp_ind |= JMP4_FLAG;
                }
				// -------------------------------------------------------------------------
			}
			
			//store jump index to the list
			jumps_list[step_ind] = jmp_list_ind | DP_FLAG; //mark as DP
			jumps_list[step_ind + STEP_CNT] = jmp_ind;
			
			//store next point to L2 cache
			Copy_int4_x2(L2x_cur + group * 4, x);
			Copy_int4_x2(L2y_cur + group * 4, y);

			//TPointPriv_Add_cuda(x, y, priv, jmp_x, jmp_y, inverse);
			MulPnt_cuda(x, y, jmp_x, jmp_y, inverse);
		} //group

		__syncthreads();
		//save inv from LDS to L2
		for (int group = 0; group < PNT_GROUP_CNT; group++)
		{
			SAVE_VAL_256(L2inv_cur + group * 4, LDS_inv + group * 4, 0);
		}
		__syncthreads();
	}
	Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
}

#endif //OLD_GPU

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef OLD_GPU

//this kernel performs main jumps (Old GPU optimization path)
__global__ void KernelA(const TKparams Kparams)
{
	u64* LDS_x = (u64*)LDS;
	u64* LDS_y = (u64*)(LDS + 4 * JMP_CNT);
	u64* LDS_inv = (u64*)(LDS_y + 4 * JMP_CNT);
	u64* jmp1_table = LDS_x;

	int global_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	u64* x_kangs = Kparams.Kangs + global_ind * 8; //x
	u64* y_kangs = x_kangs + 4; //y
	u64* priv_kangs = x_kangs + 8; //priv

	u32* jumps_list = (u32*)Kparams.JumpsList + global_ind * STEP_CNT;

	//load jumps1 to LDS
	for (int i = THREAD_X; i < JMP_CNT * 4; i += BLOCK_SIZE)
	{
		jmp1_table[i] = ((u64*)Kparams.Jumps1)[i];
		LDS_y[i] = ((u64*)Kparams.Jumps1)[i + JMP_CNT * 4];
	}
	__syncthreads();

	u64 x[4], y[4], priv[4];
	u64 x0[4], y0[4];

	u64 L1S2 = ((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X];
    u32 thread_base_kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);

    // --- JUMP TABLE SELECTION FOR 4 KANGAROO GROUPS ---
    const u32 GRP_QUARTER_SIZE = Kparams.KangCnt / 4; 
    u32 thread_group_quarter = thread_base_kang_ind / GRP_QUARTER_SIZE;

    u64* Jumps_P; // Primary Jump Table (J1, J2, J3, J4)
    u64* Jumps_A; // Alternative Jump Table (J2 for TAME, J1 for WILDs)
    
    // Determine Jumps_P and Jumps_A pointers based on kangaroo type (0, 1, 2, or 3)
    if (thread_group_quarter == TAME) { // 0: Uses J1 (Jumps1) primarily
        Jumps_P = jmp1_table; 
        Jumps_A = jmp2_table; 
    } else if (thread_group_quarter == WILD1) { // 1: Uses J2 (jmp2_table) primarily
        Jumps_P = jmp2_table;
        Jumps_A = jmp1_table; 
    } else if (thread_group_quarter == WILD2) { // 2: Uses J3 (Kparams.Jumps3) primarily
        Jumps_P = (u64*)Kparams.Jumps3; 
        Jumps_A = jmp1_table;
    } else { // WILD3 (3): Uses J4 (Kparams.Jumps4) primarily
        Jumps_P = (u64*)Kparams.Jumps4;
        Jumps_A = jmp1_table;
    }
    // -----------------------------------------------------

	u64* jmp_table;
	__align__(16) u64 inverse[5];

	//load inv from global
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		Copy_int4_x2(LDS_inv + group * 4, Kparams.LastPnts + (global_ind + group * BLOCK_SIZE * BLOCK_CNT) * 8 + 4);
	}
	__syncthreads();

	//preparations (first calc for inv)
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		Copy_int4_x2(x, x_kangs + group * 4);
		Copy_int4_x2(y, y_kangs + group * 4);
		Copy_int4_x2(priv, priv_kangs + group * 4);

		u32 jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> group) & 1) ? Jumps_A : Jumps_P; // Use determined tables
		
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);

		TPointPriv_Add_cuda(x, y, priv, jmp_x, jmp_y, inverse);

		Copy_int4_x2(LDS_inv + group * 4, x); //store x0 to LDS
	}
	__syncthreads();


	//main loop
	u64* x_kangs_next = Kparams.LastPnts + global_ind * 8; //x
	u64* y_kangs_next = x_kangs_next + 4; //y
	
	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
	{
		int g_end = step_ind < 1 ? PNT_GROUP_CNT : -1;
		int group = step_ind < 1 ? 0 : PNT_GROUP_CNT - 1;

		while (group != g_end)
		{
			//load next point from global
			Copy_int4_x2(x, x_kangs + group * 4);
			Copy_int4_x2(y, y_kangs + group * 4);
			Copy_int4_x2(priv, priv_kangs + group * 4);

			//calc next point (x0, y0) from current point (x, y)
			Copy_int4_x2(x0, LDS_inv + group * 4);
			TPointPriv_Add_cuda(x, y, priv, x0, LDS_inv + group * 4, inverse);

			//save point for next step (x, y)
			Copy_int4_x2(x_kangs_next + group * 4, x); 
			Copy_int4_x2(y_kangs_next + group * 4, y);

			//calc jmp index
			u32 jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? Jumps_A : Jumps_P; // Use determined tables
			
			__align__(16) u64 jmp_x[4];
			__align__(16) u64 jmp_y[4];
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			
			u32 jmp_list_ind = jmp_ind;

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1ull << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1ull << group);
				// --- Set correct flag based on Primary Jump Table used when loop broke ---
                if (thread_group_quarter == TAME || thread_group_quarter == WILD1) {
                    jmp_ind |= JMP2_FLAG;
                } else if (thread_group_quarter == WILD2) {
                    jmp_ind |= JMP3_FLAG; 
                } else { // WILD3
                    jmp_ind |= JMP4_FLAG;
                }
				// -------------------------------------------------------------------------
			}
			
			//store jump index to the list
			jumps_list[step_ind] = jmp_list_ind | DP_FLAG; //mark as DP
			jumps_list[step_ind + STEP_CNT] = jmp_ind;
			
			//TPointPriv_Add_cuda(x, y, priv, jmp_x, jmp_y, inverse);
			MulPnt_cuda(x, y, jmp_x, jmp_y, inverse);

			//preps to calc next inv
			Copy_int4_x2(LDS_inv + group * 4, x);

			group += step_ind < 1 ? 1 : -1;
		} //group
		__syncthreads();
		if (step_ind < 1)
		{
			x_kangs = Kparams.LastPnts + global_ind * 8;
			y_kangs = x_kangs + 4;
			x_kangs_next = Kparams.Kangs + global_ind * 8;
			y_kangs_next = x_kangs_next + 4;
		}
		else
		{
			x_kangs = x_kangs_next;
			y_kangs = y_kangs_next;
			if (step_ind & 1)
			{
				x_kangs_next = Kparams.Kangs + global_ind * 8;
				y_kangs_next = x_kangs_next + 4;
			}
			else
			{
				x_kangs_next = Kparams.LastPnts + global_ind * 8;
				y_kangs_next = x_kangs_next + 4;
			}
		}
		__syncthreads();
	}
	((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
}

#endif //OLD_GPU

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//this function makes final distinguished point output to the global memory
extern __device__ __forceinline__ void BuildDP(const TKparams& Kparams, int kang_ind, u64* d)
{
	if (Kparams.DPs_out)
	{
		u32 DPs[12];
		DPs[0] = kang_ind; //index to the point
		//DPs[1] is a counter

		u64* priv = Kparams.Kangs + kang_ind * 12 + 8;
		
		u64* DPs_out = Kparams.DPs_out;
		u32* DPs_cnt = (u32*)DPs_out;

		int counter = atomicAdd(DPs_cnt, 1);
		if (counter >= MAX_DP_CNT)
			return;

		DPs_out += 4 + counter * 4; //skip counter, 4*32=128bytes
		
		*(u64*)&DPs[1] = priv[0];
		*(u64*)&DPs[2] = priv[1];
		*(u64*)&DPs[3] = priv[2];

		*(int4*)&DPs[4] = ((int4*)d)[0];
		*(u64*)&DPs[8] = d[2];
		
        // --- Updated: Kangaroo type calculation (0, 1, 2, or 3) ---
		// Divides the total kangaroo count into 4 equal quarters for TAME, WILD1, WILD2, WILD3
		DPs[10] = 4 * kang_ind / Kparams.KangCnt; //kang type 
        // -----------------------------------------------------------

		DPs[11] = Kparams.BlockSize * Kparams.BlockCnt * PNT_GROUP_CNT; //full range size
		
		*(int4*)DPs_out = *(int4*)&DPs[0];
		*(int4*)(DPs_out + 4) = *(int4*)&DPs[4];
		*(int4*)(DPs_out + 8) = *(int4*)&DPs[8];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Placeholder kernels/functions for structural completeness (assuming they exist in your source)
#ifndef OLD_GPU
extern "C" __global__ __launch_bounds__(512, 2) void KernelB(const TKparams Kparams) { /* ... kernel logic ... */ }
extern "C" __global__ __launch_bounds__(512, 2) void KernelC(const TKparams Kparams) { /* ... kernel logic ... */ }
#else
__global__ void KernelB(const TKparams Kparams) { /* ... kernel logic ... */ }
__global__ void KernelC(const TKparams Kparams) { /* ... kernel logic ... */ }
#endif
__global__ void KernelGen(const TKparams Kparams) { /* ... kernel logic ... */ }


void CallGpuKernelABC(TKparams Kparams)
{
	KernelA <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelA_LDS_Size >>> (Kparams);
	KernelB <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelB_LDS_Size >>> (Kparams);
	KernelC <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelC_LDS_Size >>> (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
	KernelGen << < Kparams.BlockCnt, Kparams.BlockSize, 0 >> > (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
	cudaError_t err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
	if (err != cudaSuccess)
		return err;

	err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
	if (err != cudaSuccess)
		return err;

	err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
	if (err != cudaSuccess)
		return err;

	err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, 8 * JMP_CNT * 8);
	if (err != cudaSuccess)
		return err;

	return cudaSuccess;
}
