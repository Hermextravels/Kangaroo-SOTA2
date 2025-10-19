## Host C++ compiler (respect env override)
CXX ?= g++

## Prefer explicit CUDA_PATH, then CUDA_HOME, else fallback to /usr/local/cuda
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

## Select a GCC 12 host compiler for nvcc when available (Ubuntu 24.04 ships GCC 13)
## You can override via: make NVCC_CCBIN=/usr/bin/g++-12
NVCC_CCBIN ?= $(shell if command -v g++-12 >/dev/null 2>&1; then command -v g++-12; else command -v $(CC); fi)

## Optional: allow unsupported compiler (set to 1 to append flag)
ALLOW_UNSUPPORTED ?= 0
ALLOW_UNSUPPORTED_FLAG := $(if $(filter 1,$(ALLOW_UNSUPPORTED)),-allow-unsupported-compiler,)

## Compiler flags
CXXFLAGS := -O3 -I$(CUDA_ROOT)/include
NVCCFLAGS := -O3 -ccbin=$(NVCC_CCBIN) $(ALLOW_UNSUPPORTED_FLAG) \
	-gencode=arch=compute_89,code=compute_89 \
	-gencode=arch=compute_86,code=compute_86 \
	-gencode=arch=compute_75,code=compute_75 \
	-gencode=arch=compute_61,code=compute_61
LDFLAGS := -L$(CUDA_ROOT)/lib64 -lcudart -pthread -lm

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
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS)
