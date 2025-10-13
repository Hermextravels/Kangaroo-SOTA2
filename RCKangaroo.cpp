// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#define RESUME_FILE_NAME "RESUME_KEY.TXT"

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];
EcJMP EcJumps4[JMP_CNT]; // New: Fourth jump table

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
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

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

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
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
#define WILD_JUMP_START EcJumps1[1].dist.Set(10); EcJumps2[1].dist.Set(10); EcJumps3[1].dist.Set(10); EcJumps4[1].dist.Set(10);
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_found)
{
	//check if solution found
	if (gSolved)
		return true;

	TotalOps = 0;
	PntIndex = 0;
	gPntToSolve.Assign(PntToSolve);
	PntKeyFound.SetZero();
	PntTotalOps = 0;

	//check if we found the key in last run
	if (!db.FindPoint(PntToSolve, NULL, &PntKeyFound, NULL))
	{
		PntKeyFound.SetZero();
	}
	else
	{
		if (pk_found)
			pk_found->Assign(PntKeyFound);
		return true;
	}

	//init kangs
	//set half range
	EcInt one;
	one.Set(1);
	Int_HalfRange = g_N;
	Int_HalfRange.Add(one);
	Int_HalfRange.ShiftRight(1);

	Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
	Pnt_NegHalfRange = ec.NegPoint(Pnt_HalfRange);
	Int_TameOffset.Set(g_N.data[0] / 3);

	//init wild jumps
	u64 seed = 0x9599B29944A9D3F8;
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcInt jump_dist;
		jump_dist.Set(1);

		EcPoint p = ec.MultiplyG(jump_dist);

		EcJumps1[i].p.Assign(p);
		EcJumps1[i].dist.Assign(jump_dist);

		EcJumps2[i].p.Assign(p);
		EcJumps2[i].dist.Assign(jump_dist);

		EcJumps3[i].p.Assign(p);
		EcJumps3[i].dist.Assign(jump_dist);
		
		EcJumps4[i].p.Assign(p); // ADDED: Initialization for the fourth jump table
		EcJumps4[i].dist.Assign(jump_dist);
	}

	//do the rest of jump table init here
	WILD_JUMP_START;

	//start new search
	for (int i = 0; i < GpuCnt; i++)
	{
		// FIX: Passing the newly declared EcJumps4 array as the 7th argument
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3, EcJumps4)) 
		{
			printf("GPU %d, Prepare failed\\r\\n", i);
			return false;
		}
	}

	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Start();

	//wait for thread end
	while (ThrCnt > 0)
	{
		Sleep(100);
	}

	if (gSolved)
	{
		if (pk_found)
			pk_found->Assign(IntSolvedKey);
		return true;
	}

	return false;
}

void PrintHelp()
{
	printf("Usage: rckangaroo [option...]\r\n");
	printf("Options:\r\n");
	printf("  -g - Tames generation mode, generates tames and saves to dptables\r\n");
	printf("  -dp XX - DP size in bits, default is 16\r\n");
	printf("  -r XX - Range size in bits, default is 78\r\n");
	printf("  -solve KEY - solves given key (in hex format, e.g. -solve 1)\r\n");
	printf("  -max_gpu XX - max GPU to use, default is all\r\n");
	printf("  -benchmark - benchmark mode, generates and solves random keys\r\n");
	printf("\r\n");
}

int main(int argc, char** argv)
{
	//init GPU kangs
	cudaError_t err = cudaGetDeviceCount(&GpuCnt);
	if (err != cudaSuccess)
		GpuCnt = 0;

	if (!GpuCnt)
		printf("No CUDA device found\\r\\n");

	if (GpuCnt > MAX_GPU_CNT)
		GpuCnt = MAX_GPU_CNT;

	printf("Found %d CUDA devices\\r\n", GpuCnt);

	IntSolvedKey.SetZero();

	for (int i = 0; i < GpuCnt; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("GPU %d: %s\r\n", i, prop.name);
		GpuKangs[i] = new RCGpuKang(i, prop.multiProcessorCount);
	}

	//parse cmdline
	char KeyStr[1000] = { 0 };
	gGenMode = false;
	gRange = 0;
	gDP = 0;
	bool bSolve = false;
	bool bBenchmark = false;
	
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-g") == 0)
			gGenMode = true;
		else if (strcmp(argv[i], "-solve") == 0)
		{
			bSolve = true;
			i++;
			if (i < argc)
				strcpy(KeyStr, argv[i]);
			else
			{
				PrintHelp();
				goto label_end;
			}
		}
		else if (strcmp(argv[i], "-r") == 0)
		{
			i++;
			if (i < argc)
				gRange = atoi(argv[i]);
			else
			{
				PrintHelp();
				goto label_end;
			}
		}
		else if (strcmp(argv[i], "-dp") == 0)
		{
			i++;
			if (i < argc)
				gDP = atoi(argv[i]);
			else
			{
				PrintHelp();
				goto label_end;
			}
		}
		else if (strcmp(argv[i], "-max_gpu") == 0)
		{
			i++;
			if (i < argc)
			{
				int max_gpu = atoi(argv[i]);
				if (max_gpu < GpuCnt)
					GpuCnt = max_gpu;
			}
			else
			{
				PrintHelp();
				goto label_end;
			}
		}
		else if (strcmp(argv[i], "-benchmark") == 0)
			bBenchmark = true;
		else
		{
			PrintHelp();
			goto label_end;
		}
	}

	if (bSolve)
	{
		//solve key
		EcInt pk_found;
		if (!Int_TameOffset.SetHexStr(KeyStr))
		{
			printf("Invalid key format\r\n");
			goto label_end;
		}

		if (!gRange)
			gRange = 78;
		if (!gDP)
			gDP = 16;
		
		PntToSolve = ec.MultiplyG(Int_TameOffset);

		if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
		{
			printf("\r\nKey not found, try increasing range or DP value\r\n");
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
				printf("\r\nKey not found, try increasing range or DP value\r\n");
				goto label_end;
			}
			
			//check if pk is correct
			EcPoint p = ec.MultiplyG(pk_found);
			if (p.IsEqual(PntToSolve))
				printf("Correct, private key found\r\n");
			else
			{
				printf("Error: private key is incorrect\r\n");
				printf("Expected:");
				pk.Print();
				printf("Found:");
				pk_found.Print();
				PntToSolve.Print();
				p.Print();
				printf("\r\n");
			}
		}
	}

label_end:
	//cleanup
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];

	return 0;
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
			strcpy(gTamesFileName, argv[ci]);
			ci++;
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
	printf("* RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
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

	InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE - SUBRANGE SEARCH with Auto-Resume\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk_found;

		// 1. Define the full range bounds for Puzzle #135
		// Key Range: 4000000000000000000000000000000000:7fffffffffffffffffffffffffffffffff
		
		EcInt FullRangeEnd;
		FullRangeEnd.SetHexStr("7fffffffffffffffffffffffffffffffff");
		
		EcInt DefaultStart; // The absolute start of the puzzle's range (2^134)
		DefaultStart.SetHexStr("4000000000000000000000000000000000");

		// *** FIX for subtraction error: define EcInt for value 1 ***
		EcInt IntOne;
		IntOne.Set(1);

		EcInt CurrentStart;
		
		// **********************************************
		// AUTO-RESUME LOGIC: Read last saved key
		// **********************************************
		char resume_hex[100] = { 0 };
		FILE* fp_resume = fopen(RESUME_FILE_NAME, "r");
		if (fp_resume)
		{
			if (fscanf(fp_resume, "%s", resume_hex) == 1)
			{
				CurrentStart.SetHexStr(resume_hex);
				printf("Resuming from key found in %s: %s\r\n", RESUME_FILE_NAME, resume_hex);
			}
			fclose(fp_resume);
		}
		
		// Priority: Command Line (-start) > Resume File > Puzzle Default
		if (gStartSet)
		{
			CurrentStart = gStart;
			printf("Command line -start argument overrides resume file.\r\n");
		}
		else if (CurrentStart.IsZero())
		{
			CurrentStart = DefaultStart;
			printf("Starting from puzzle's default key: %s\r\n", "4000000000000000000000000000000000");
		}
		// **********************************************
		
		// Check if the necessary range parameter is set
		if (gRange == 0)
		{
			printf("error: You must set a small subrange size using -range, e.g., -range 35\r\n");
			goto label_end;
		}
		const int SUBRANGE_BITS = gRange;
		
		// Calculate the increment step (2^SUBRANGE_BITS)
		EcInt RangeIncrement;
		RangeIncrement.Set(1);
		RangeIncrement.ShiftLeft(SUBRANGE_BITS);
		
		int CurrentChunk = 0;

		char sx[100];
		gPubKey.x.GetHexStr(sx);
		printf("Target PubKey X: %s\r\n", sx);
		gPubKey.y.GetHexStr(sx);
		printf("Target PubKey Y: %s\r\n", sx);


		// **********************************************
		// START SUBRANGE SEARCH LOOP
		// **********************************************
		// FIX: Use IsLessThanU for comparison
		while (CurrentStart.IsLessThanU(FullRangeEnd))
		{
			EcInt CurrentEnd = CurrentStart;
			CurrentEnd.Add(RangeIncrement);
			
			// FIX: Subtract the EcInt 'IntOne'
			CurrentEnd.Sub(IntOne); 
			
			// Clip the end if it exceeds the final range end.
			// FIX: Check if (FullRangeEnd < CurrentEnd) which means (CurrentEnd > FullRangeEnd)
			if (FullRangeEnd.IsLessThanU(CurrentEnd)) 
			{
				CurrentEnd = FullRangeEnd;
			}
			
			// --- Set up the search parameters for this chunk ---
			gStart = CurrentStart;
			
			// PntToSolve becomes P - gStart*G
			PntToSolve = gPubKey;
			PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP(); // PntOfs = -gStart*G
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs); // PntToSolve = P - gStart*G
			
			
			printf("\r\n================================================================================\r\n");
			gStart.GetHexStr(sx);
			printf("CHUNK %d: Searching Range %d bits\r\n  Offset (Start): %s\r\n", CurrentChunk, SUBRANGE_BITS, sx);
			CurrentEnd.GetHexStr(sx);
			printf("  End: %s\r\n", sx);
			printf("================================================================================\r\n");
			
			// **********************************************
			// AUTO-RESUME LOGIC: Save the NEXT start key
			// **********************************************
			EcInt NextStart = CurrentStart;
			NextStart.Add(RangeIncrement);

			// Write the NEXT chunk's start key to the resume file
			FILE* fp_save = fopen(RESUME_FILE_NAME, "w");
			if (fp_save)
			{
				NextStart.GetHexStr(resume_hex);
				fprintf(fp_save, "%s", resume_hex);
				fclose(fp_save);
			} else {
				printf("WARNING: Could not write to resume file %s. Auto-resumption will fail on crash.\r\n", RESUME_FILE_NAME);
			}
			// **********************************************


			// Call SolvePoint with the small range and offset
			if (SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				// Collision found!
				pk_found.Add(gStart); // Calculate the final key (relative + offset)
				
				// Delete the resume file on success
				remove(RESUME_FILE_NAME); 

				goto label_solved; 
			}
			
			if (gIsOpsLimit)
			{
				// Ops limit was hit. The resume file is already updated with the NEXT chunk's start.
				printf("Operation limit hit. Resume file %s updated. Re-run to continue.\r\n", RESUME_FILE_NAME);
				goto label_end;
			}
			
			// Move to the next chunk start
			CurrentStart.Add(RangeIncrement);
			CurrentChunk++;
		}
		// **********************************************
		// END SUBRANGE SEARCH LOOP
		// **********************************************
		
		printf("\r\nSEARCH COMPLETE: Key not found in the full range. Deleting resume file.\r\n");
		remove(RESUME_FILE_NAME);
		goto label_end;
		

	// The original code's "happy end" logic now becomes a target label.
	label_solved: ;
		// The key is already the absolute key: pk_found = relative_key + gStart

		// Verify the final key
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
}