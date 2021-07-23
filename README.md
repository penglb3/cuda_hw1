# CUDA Homework 1 Full Code
P's CUDA Homework 1 Full Code [Public Version]

## Requirements
For CUDA: 
- nvcc

For OpenMP:
- GCC (presumably, because MSVC is stuck with OpenMP 2.0!) 

(Theoretically that's all you need, because 1. nvcc will choose its base compiler 2. GCC has native support for OpenMP 3.1+)

## Compile
For CUDA:
- Linux(GCC): `nvcc matrixAdd_cuda.cu -o matrixAdd_cuda -Xcompiler -fopenmp`
- Windows(MSVC): `nvcc matrixAdd_cuda.cu -o matrixAdd_cuda.exe -Xcompiler /openmp`

For OpenMP:
- Linux(GCC): `g++ matrixAdd_omp.cpp -o matrixAdd_omp -std=c++11 -fopenmp`
- Windows(MinGW): `g++ matrixAdd_omp.cpp -o matrixAdd_omp.exe -std=c++11 -fopenmp`

## Program arguments
**Note that all arguments have default values, which means they are all optional.**

For CUDA (`matrixAdd_cuda`)
- `-r [rows of matrix]`
- `-c [columns of matrix]`
- `-b [threads per block]`
- `-x [threads along x axis per block]`
- `-y [threads along y axis per block]`
- `-N [elements each thread is dealing with]`

For OpenMP (`matrixAdd_omp`)
- `-r [rows of matrix]`
- `-c [columns of matrix]`
- `-t [number of CPU threads]`
- `-a [time to repeat]`

## Note
The code is tested under Windows Environment, so I can't guarantee it will compile or run without problem under Linux. 
