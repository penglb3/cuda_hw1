# cuda_hw1
My CUDA Homework 1 Full Code [For TA and teachers]

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
- Linux(GCC): `g++ matrixAdd_omp.cpp -o matrixAdd_omp -fopenmp`
- Windows(MinGW): `g++ matrixAdd_omp.cpp -o matrixAdd_omp.exe -fopenmp`

## Program arguments
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
