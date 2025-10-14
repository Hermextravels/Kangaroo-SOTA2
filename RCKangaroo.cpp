// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;
u32 gDPTableSlotsOverride = 0; // 0 = auto, otherwise override slots per kangaroo
bool gDPAuto = false; // auto-select DP bits based on memory and range
u32 gMemCapMB = 3800; // target DP table memory cap in MB (default ~3.8GB)
u32 gSplitRangeN = 0; // optional: generate N subranges and exit

// Checkpointing: host buffer for all kangaroo states
u8* pKangsState = NULL;
u64 KangsStateSize = 0; // Total size in bytes
bool gResumeOk = false;
// New: resume/main-mode support and configurable checkpoint interval
bool gUseResume = false; // set by -resume
u32 gCheckpointIntervalSecs = CHECKPOINT_INTERVAL_SECS;

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	snprintf(drvver, sizeof(drvver), "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		// Report free/total memory on this GPU
		size_t freeB = 0, totalB = 0;
		if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
		{
			printf("GPU %d memory: free %.2f GiB / total %.2f GiB\r\n", i,
				(double)freeB / (1024.0 * 1024.0 * 1024.0), (double)totalB / (1024.0 * 1024.0 * 1024.0));
		}

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
            GpuKangs[GpuCnt]->KangsPreloaded = false;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];

		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}

			bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
			if (!res)
			{
				bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
				if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
					;// ToLog("W1 and W2 collides in mirror");
				else
				{
					printf("Collision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}
			gSolved = true;
			break;
		}
	}
}

// Save all kangaroo states (gather from all GPUs and write to file)
bool SaveKangStates()
{
	if (KangsStateSize == 0 || !pKangsState) return false;
	u64 cur_pos = 0;
	for (int i = 0; i < GpuCnt; i++)
	{
	size_t size = (size_t)GpuKangs[i]->KangCnt * 12 * sizeof(u64);
		// synchronize GPU to reduce chance of copying mid-kernel
		cudaSetDevice(GpuKangs[i]->CudaIndex);
		cudaDeviceSynchronize();
		if (!GpuKangs[i]->SaveKangs(pKangsState + cur_pos))
			return false;
		cur_pos += size;
	}

	FILE* fp = fopen(KANGS_RESUME_FILE, "wb");
	if (!fp)
	{
		printf("WARNING: Cannot open %s for saving!\r\n", KANGS_RESUME_FILE);
		return false;
	}
	// header: total size and GPU count
	u64 header[2] = {KangsStateSize, (u64)GpuCnt};
	fwrite(header, 1, sizeof(header), fp);
	fwrite(pKangsState, 1, (size_t)KangsStateSize, fp);
	fclose(fp);
	return true;
}

// Load all kangaroo states from file and restore them to device memory
bool LoadKangStates()
{
	FILE* fp = fopen(KANGS_RESUME_FILE, "rb");
	if (!fp)
	{
		printf("Info: Resume file %s not found. Starting a new search.\r\n", KANGS_RESUME_FILE);
		return false;
	}
	u64 header[2];
	if (fread(header, 1, sizeof(header), fp) != sizeof(header))
	{
		printf("Error reading header from %s\r\n", KANGS_RESUME_FILE);
		fclose(fp);
		return false;
	}
	u64 file_size = header[0];
	u32 file_gpu_cnt = (u32)header[1];
	if (file_size != KangsStateSize || file_gpu_cnt != GpuCnt)
	{
		printf("Error: Resume file configuration mismatch. Starting a new search.\r\n");
		printf("File Size: %llu, Current Size: %llu\r\n", file_size, KangsStateSize);
		printf("File GpuCnt: %u, Current GpuCnt: %u\r\n", file_gpu_cnt, GpuCnt);
		fclose(fp);
		return false;
	}
	if (fread(pKangsState, 1, (size_t)KangsStateSize, fp) != (size_t)KangsStateSize)
	{
		printf("Error reading data from %s\r\n", KANGS_RESUME_FILE);
		fclose(fp);
		return false;
	}
	fclose(fp);

	u64 cur_pos = 0;
	for (int i = 0; i < GpuCnt; i++)
	{
	size_t size = (size_t)GpuKangs[i]->KangCnt * 12 * sizeof(u64);
		// synchronize device before loading
		cudaSetDevice(GpuKangs[i]->CudaIndex);
		cudaDeviceSynchronize();
		if (!GpuKangs[i]->LoadKangs(pKangsState + cur_pos))
			return false;
		cur_pos += size;
	}
	printf("Successfully resumed from %s\r\n", KANGS_RESUME_FILE);
	return true;
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

	u64 sec = (GetTickCount64() - tm_start) / 1000;
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec - days * (3600 * 24)) / 3600;
	int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
	 
	printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n", gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "), speed, gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000, days, hours, min, exp_days, exp_hours, exp_min);
}

// Compute an auto DP value (and optionally DP slots) based on available memory and desired cap.
// We do this at host side before Prepare() so all GPUs use a consistent DP and fit in the cap.
// Heuristic: choose DP bits such that expected DPs per kangaroo per iteration is small (<= ~2),
// and choose slots 6..12 that fit within the DP memory cap across GPUs. If the cap is tight, raise DP.
static void AutoSelectDPAndSlots(int& outDP, u32& outSlots)
{
	// Estimate OPS ~ 1.15 * 2^(range/2)
	double ops = 1.15 * pow(2.0, gRange / 2.0);
	// Aggregate worst-case across GPUs for KangCnt and free memory
	u64 minFree = (u64)-1;
	int maxKang = 0;
	for (int i = 0; i < GpuCnt; i++)
	{
		int blockCnt = GpuKangs[i]->mpCnt;
		int blockSize = GpuKangs[i]->IsOldGpu ? 512 : 256;
		int groupCnt = GpuKangs[i]->IsOldGpu ? 64 : 24;
		int kangCnt = blockCnt * blockSize * groupCnt;
		if (kangCnt > maxKang) maxKang = kangCnt;
		// query free mem on this GPU
		size_t freeB = 0, totalB = 0;
		cudaSetDevice(GpuKangs[i]->CudaIndex);
		if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
		{
			if ((u64)freeB < minFree) minFree = (u64)freeB;
		}
	}
	if (maxKang <= 0) { outDP = 20; outSlots = 8; return; }
	// Budget for DP table in bytes: min(user cap, actual free - safety)
	u64 capB = (u64)gMemCapMB * 1024ull * 1024ull;
	if (minFree != (u64)-1)
	{
		u64 safeFree = (minFree > 512ull * 1024ull * 1024ull) ? (minFree - 512ull * 1024ull * 1024ull) : (minFree / 2);
		if (capB > safeFree) capB = safeFree;
	}

	// Try DP bits from 16..40: higher DP -> fewer DPs emitted. Choose slots 4..16 to fit cap.
	int bestDP = 22; u32 bestSlots = 8;
	for (int dp = 16; dp <= 40; dp++)
	{
		double dp_val = (double)(1ull << dp);
		double path_per_kang = ops / (double)maxKang;
		double dps_per_kang = path_per_kang / dp_val;
		// desired slots  = small multiple of expected per-kang DPs
		u32 slots = (u32)std::max(4.0, std::min(16.0, floor(dps_per_kang * 4.0) + 6.0));
		// Estimate DP table bytes = KangCnt * (16 * slots + sizeof(u32))
		u64 bytes = (u64)maxKang * (16ull * (u64)slots + sizeof(u32));
		if (bytes <= capB)
		{
			bestDP = dp; bestSlots = slots;
			// Prefer lower DP (more DPs) until memory cap would be exceeded; break when we hit ~2 DPs per kang
			if (dps_per_kang <= 2.0) break;
		}
		else
		{
			// If memory cap is exceeded, raise DP further
			continue;
		}
	}
	outDP = std::max(20, std::min(40, bestDP));
	outSlots = std::max(4u, std::min(16u, bestSlots));
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 60)) 
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}

	u64 total_kangs = GpuKangs[0]->CalcKangCnt();
	for (int i = 1; i < GpuCnt; i++)
		total_kangs += GpuKangs[i]->CalcKangCnt();
	double path_single_kang = ops / total_kangs;	
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			printf("tames loaded\r\n");
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clear\r\n");
				db.Clear();
			}
		}
		else
			printf("tames loading failed\r\n");
	}

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
	Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
	for (int i = 0; i < GpuCnt; i++)
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
		{
			GpuKangs[i]->Failed = true;
			printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
		}

	// Allocate host buffer for checkpointing now that KangCnt is known
	if (KangsStateSize == 0)
	{
		for (int i = 0; i < GpuCnt; i++)
						KangsStateSize += (u64)GpuKangs[i]->KangCnt * 12 * sizeof(u64);
		if (KangsStateSize)
		{
			pKangsState = (u8*)malloc((size_t)KangsStateSize);
			if (!pKangsState)
			{
				printf("Memory allocation failed for pKangsState!\r\n");
				KangsStateSize = 0;
			}
		}
	}

	// If in generation mode or resume requested and we have a resume buffer, try to load previous state now
	if ((gGenMode || gUseResume) && pKangsState)
	{
		// Attempt to load; this will copy data into device buffers allocated during Prepare
		if (LoadKangStates())
		{
			gResumeOk = true;
			for (int i = 0; i < GpuCnt; i++)
				GpuKangs[i]->KangsPreloaded = true;
		}
		else
		{
			gResumeOk = false;
			for (int i = 0; i < GpuCnt; i++)
				GpuKangs[i]->KangsPreloaded = false;
		}
	}

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
    u64 last_checkpoint_time = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);
			tm_stats = GetTickCount64();
		}

		// Periodic checkpointing (enabled when resume is requested or in gen mode)
		if ((gGenMode || gUseResume) && pKangsState)
		{
			u64 current_time = GetTickCount64();
			if ((current_time - last_checkpoint_time) > ((u64)gCheckpointIntervalSecs * 1000ull))
			{
				if (SaveKangStates())
				{
					printf("Checkpoint saved to %s\r\n", KANGS_RESUME_FILE);
					last_checkpoint_time = current_time;
				}
				else
				{
					printf("Warning: checkpoint save failed\r\n");
				}
			}
		}

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("Operations limit reached\r\n");
			break;
		}
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}

	if (gIsOpsLimit)
	{
		if (gGenMode)
		{
			printf("saving tames...\r\n");
			db.Header[0] = gRange; 
			if (db.SaveToFile(gTamesFileName))
				printf("tames saved\r\n");
			else
				printf("tames saving failed\r\n");
		}
		db.Clear();
		return false;
	}

	double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			snprintf(gTamesFileName, sizeof(gTamesFileName), "%s", argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-resume") == 0)
		{
			gUseResume = true;
		}
		else
		if (strcmp(argument, "-dp-slots") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if (val < 1 || val > DPTABLE_MAX_CNT)
			{
				printf("error: invalid value for -dp-slots option (1..%d)\r\n", DPTABLE_MAX_CNT);
				return false;
			}
			gDPTableSlotsOverride = (u32)val;
		}
		else
		if (strcmp(argument, "-checkpoint-secs") == 0)
		{
			int v = atoi(argv[ci]);
			ci++;
			if (v <= 0) { printf("error: invalid value for -checkpoint-secs\r\n"); return false; }
			gCheckpointIntervalSecs = (u32)v;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
		else if (strcmp(argument, "-dp-auto") == 0)
		{
			gDPAuto = true;
		}
		else if (strcmp(argument, "-autotune") == 0)
		{
			gDPAuto = true; // alias
		}
		else if (strcmp(argument, "-mem-cap-mb") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if (val < 512 || val > 15000) { printf("error: invalid value for -mem-cap-mb option (512..15000)\r\n"); return false; }
			gMemCapMB = (u32)val;
		}
		else if (strcmp(argument, "-split-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if (val < 1 || val > 1000000) { printf("error: invalid value for -split-range option (1..1000000)\r\n"); return false; }
			gSplitRangeN = (u32)val;
		}
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	gDP = 0;
	gRange = 0;
	gStartSet = false;
	gTamesFileName[0] = 0;
	gMax = 0.0;
	gGenMode = false;
	gIsOpsLimit = false;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	if (!ParseCommandLine(argc, argv))
		return 0;

	// Determine bench mode early for any pre-init helpers
	IsBench = gPubKey.x.IsZero();
	InitGpus();

	// If user requested subrange splitting, emit N subranges and exit
	if (gSplitRangeN > 0)
	{
		if (!gStartSet || !gRange)
		{
			printf("error: -split-range requires -start and -range\r\n");
			return 0;
		}
		// Compute approximate window size and list; calculation is informational only
		u32 bits = gRange;
		u32 win = bits / gSplitRangeN ? (bits / gSplitRangeN) : bits; // naive split; prefer rounding to >=32
		if (win < 32) win = 32;
		printf("Split %u into windows of %u bits (approx).\r\n", bits, win);
		for (u32 i = 0; i < gSplitRangeN; i++)
		{
			// (We don’t compute big-int starts here; print indices and win so a wrapper can compute starts.)
			printf("window %u: i=%u, win_bits=%u\r\n", i + 1, i, win);
		}
		return 0;
	}

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	// Global DP auto-selection before GPU Prepare
	if (!IsBench && gDPAuto)
	{
		int autoDP; u32 autoSlots;
		AutoSelectDPAndSlots(autoDP, autoSlots);
		if (gDP == 0) gDP = autoDP; else gDP = std::max(gDP, (u32)autoDP);
		if (gDPTableSlotsOverride == 0) gDPTableSlotsOverride = autoSlots;
		// estimate memory footprint with worst-case KangCnt across GPUs
		int maxKang = 0;
		for (int i = 0; i < GpuCnt; i++)
		{
			int blockCnt = GpuKangs[i]->mpCnt;
			int blockSize = GpuKangs[i]->IsOldGpu ? 512 : 256;
			int groupCnt = GpuKangs[i]->IsOldGpu ? 64 : 24;
			int kangCnt = blockCnt * blockSize * groupCnt;
			if (kangCnt > maxKang) maxKang = kangCnt;
		}
		u64 dpBytes = (u64)maxKang * (16ull * (u64)gDPTableSlotsOverride + sizeof(u32));
		double dpGiB = ((double)dpBytes) / (1024.0 * 1024.0 * 1024.0);
		double ops = 1.15 * pow(2.0, gRange / 2.0);
		double dpsPerKang = (ops / (double)maxKang) / (double)(1ull << gDP);
		printf("Auto DP: DP=%u, slots/kang=%u, est DP memory: %.3f GiB per GPU, est DPs/kang ~ %.4f\r\n", gDP, gDPTableSlotsOverride, dpGiB, dpsPerKang);
		if (dpsPerKang < 0.05)
			printf("Note: very sparse DPs per kangaroo; consider lowering DP by 1–2 for better observability if memory allows.\r\n");
	}

	// Note: resume/load will be attempted after GPU Prepare (device memory allocation)

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
			PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

		if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
		{
			if (!gIsOpsLimit)
				printf("FATAL ERROR: SolvePoint failed\r\n");
			goto label_end;
		}
		pk_found.AddModP(gStart);
		EcPoint tmp = ec.MultiplyG(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			//generate random pk
			pk.RndBits(gRange);
			PntToSolve = ec.MultiplyG(pk);

			if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				if (!gIsOpsLimit)
					printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
	if (pKangsState)
		free(pKangsState);
}

