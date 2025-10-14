CC := g++
CUDA_PATH ?= $(CUDA_HOME)
ifeq ($(CUDA_PATH),)
CUDA_PATH := /usr/local/cuda
endif
NVCC ?= $(CUDA_PATH)/bin/nvcc

CCFLAGS := -O3 -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_61,code=compute_61
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread

.PHONY: check_cuda
check_cuda:
	@if [ ! -x "$(NVCC)" ]; then \
		echo "nvcc not found at $(NVCC)"; \
		echo "Set CUDA_PATH or CUDA_HOME to your CUDA installation, e.g. export CUDA_PATH=/usr/local/cuda"; \
		exit 1; \
	fi

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: check_cuda $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS)
