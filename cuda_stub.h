// Minimal CUDA stubs for CPU-only builds when CUDA headers are missing
#pragma once

#include <stddef.h>
#include <stdio.h>

typedef int cudaError_t;
typedef struct {
	struct {
		void* base_ptr;
		size_t num_bytes;
		double hitRatio;
		int hitProp;
		int missProp;
	} accessPolicyWindow;
} cudaStreamAttrValue;
typedef int cudaLimit;

#define cudaSuccess 0

static inline const char* cudaGetErrorString(cudaError_t) { return "no-cuda"; }
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
// Minimal device property struct
typedef struct {
	int major;
	int minor;
	int multiProcessorCount;
	char name[256];
	unsigned long long totalGlobalMem;
	int pciBusID;
	int l2CacheSize;
	int persistingL2CacheMaxSize;
} cudaDeviceProp;

static inline cudaError_t cudaGetDeviceCount(int* cnt) { if (cnt) *cnt = 0; return cudaSuccess; }
static inline cudaError_t cudaRuntimeGetVersion(int* v) { if (v) *v = 0; return cudaSuccess; }
static inline cudaError_t cudaDriverGetVersion(int* v) { if (v) *v = 0; return cudaSuccess; }
static inline cudaError_t cudaSetDeviceFlags(unsigned int) { return cudaSuccess; }
static inline cudaError_t cudaSetDeviceScheduleFlag(int) { return cudaSuccess; }

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int idx)
{
	if (!prop) return 1;
	snprintf(prop->name, sizeof(prop->name), "cpu-stub-%d", idx);
	prop->major = 1;
	prop->minor = 0;
	prop->multiProcessorCount = 1;
	prop->totalGlobalMem = 1024ULL * 1024ULL * 1024ULL; // 1GB
	prop->pciBusID = 0;
	prop->l2CacheSize = 16 * 1024 * 1024; // 16MB
	prop->persistingL2CacheMaxSize = prop->l2CacheSize;
	return cudaSuccess;
}
static inline cudaError_t cudaMalloc(void** ptr, size_t size) { *ptr = malloc(size); return (*ptr) ? cudaSuccess : 1; }
static inline cudaError_t cudaFree(void* ptr) { free(ptr); return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int) { memcpy(dst, src, count); return cudaSuccess; }
static inline cudaError_t cudaMemset(void* dst, int val, size_t count) { memset(dst, val, count); return cudaSuccess; }
static inline cudaError_t cudaDeviceSetLimit(int, size_t) { return cudaSuccess; }
static inline cudaError_t cudaStreamSetAttribute(void*, int, void*) { return cudaSuccess; }

// dummy constants used by code
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
#define cudaLimitPersistingL2CacheSize 0
#define cudaAccessPropertyPersisting 0
#define cudaAccessPropertyStreaming 1
#define cudaStreamAttributeAccessPolicyWindow 0
#define cudaDeviceScheduleBlockingSync 0
#define cudaDeviceScheduleBlockingSync 0
