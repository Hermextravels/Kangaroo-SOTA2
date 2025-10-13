CC := g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# If CUDA headers are missing, build a CPU-only variant
ifeq ($(wildcard $(CUDA_PATH)/include/cuda_runtime.h),)
    NO_CUDA := 1
endif

ifdef NO_CUDA
CCFLAGS := -O3 -I. -DNO_CUDA -march=native -mtune=native -std=gnu++11
else
CCFLAGS := -O3 -I$(CUDA_PATH)/include -I. -march=native -mtune=native -std=gnu++11
endif
NVCCFLAGS := -O3 --use_fast_math --ptxas-options="-v -dlcm=ca" \
             -gencode=arch=compute_75,code=sm_75 \
             -maxrregcount=128 \
             --extra-device-vectorization \
             --gpu-architecture=sm_75 \
             --ftz=true --prec-div=false --prec-sqrt=false \
             -Xptxas=-v,-dlcm=ca,-maxrregcount=128 \
             -DUSE_SHARED_MEM_CACHE \
             -Xcompiler "-O3 -march=native -mtune=native"
ifdef NO_CUDA
LDFLAGS := -pthread
else
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread
endif

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp globals.cpp
ifdef NO_CUDA
CPU_SRC += gpu_cpu_stubs.cpp
endif
ifdef NO_CUDA
GPU_SRC :=
else
GPU_SRC := RCGpuCore.cu
endif

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS)
