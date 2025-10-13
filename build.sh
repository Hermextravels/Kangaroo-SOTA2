#!/bin/bash

# Compile and run test script

nvcc --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA nvcc compiler not found!"
    exit 1
fi

g++ --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: g++ compiler not found!"
    exit 1
fi

make clean
make -j 4

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"