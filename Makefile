CC := g++
# Prefer explicit CUDA_PATH, then CUDA_HOME, else fallback to /usr/local/cuda
ifdef CUDA_PATH
CUDA_ROOT := $(CUDA_PATH)
else
ifdef CUDA_HOME
CUDA_ROOT := $(CUDA_HOME)
else
CUDA_ROOT := /usr/local/cuda
endif
endif

NVCC ?= $(CUDA_ROOT)/bin/nvcc

# Compiler flags
CCFLAGS := -O3 -I$(CUDA_ROOT)/include
NVCCFLAGS := -O3 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_61,code=compute_61
LDFLAGS := -L$(CUDA_ROOT)/lib64 -lcudart -pthread

.PHONY: check_cuda
check_cuda:
	@if [ ! -x "$(NVCC)" ] || [ ! -d "$(CUDA_ROOT)/include" ]; then \
		echo "CUDA toolchain not found under $(CUDA_ROOT)"; \
		echo "nvcc: $(NVCC)"; \
		echo "Include dir: $(CUDA_ROOT)/include"; \
		echo "Set CUDA_PATH or CUDA_HOME to your CUDA installation, e.g."; \
		echo "  export CUDA_PATH=/usr/local/cuda"; \
		echo "or install CUDA and ensure nvcc and headers are available."; \
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
