#!/bin/bash
# Build script optimized for NVIDIA Tesla T4 GPU
# Compute Capability 7.5, 40 SMs, 16GB GDDR6

echo "========================================="
echo "RCKangaroo - Tesla T4 Optimized Build"
echo "========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA nvcc compiler not found!"
    echo "Please install CUDA toolkit and ensure it's in your PATH"
    exit 1
fi

# Display CUDA version
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ compiler not found!"
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build with T4-optimized settings
echo ""
echo "Building with Tesla T4 optimizations..."
echo "- Compute Capability: 7.5 (Turing)"
echo "- Block Size: 512 threads"
echo "- Point Groups: 64"
echo "- Target SMs: 40"
echo "- Fast Math: Enabled"
echo "- Max Registers: 255"
echo ""

make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build successful!"
    echo "========================================="
    echo ""
    echo "Binary: ./rckangaroo"
    echo ""
    echo "Expected performance on Tesla T4:"
    echo "  ~1.5-2.0 GKeys/s (depending on bit range)"
    echo ""
    echo "Usage examples:"
    echo ""
    echo "  Benchmark mode (random keys):"
    echo "    ./rckangaroo -dp 16 -range 76"
    echo ""
    echo "  Solve specific key (puzzle #71):"
    echo "    ./rckangaroo -dp 15 -range 71 -start 400000000000000000 -pubkey <pubkey>"
    echo ""
    echo "  Generate tames for faster solving:"
    echo "    ./rckangaroo -dp 16 -range 76 -tames tames76.dat -max 10"
    echo ""
else
    echo ""
    echo "========================================="
    echo "Build FAILED!"
    echo "========================================="
    echo ""
    echo "Common issues:"
    echo "  1. CUDA toolkit not properly installed"
    echo "  2. CUDA compute capability mismatch"
    echo "  3. Missing dependencies"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi
