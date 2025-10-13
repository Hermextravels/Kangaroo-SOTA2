CC := g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc


CCFLAGS := -O3 -I$(CUDA_PATH)/include -I.
NVCCFLAGS := -O3 --use_fast_math -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_86,code=compute_86
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread

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
