CC := g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# CUDA architecture and paths
ARCH ?= sm_75
CUDA_ARCH := -gencode=arch=compute_75,code=$(ARCH)

# Compiler flags
CCFLAGS := -O3 -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/x86_64-linux/include
NVCCFLAGS := -O3 $(CUDA_ARCH) --use_fast_math -maxrregcount=255 -I$(CUDA_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -lcudart -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

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
