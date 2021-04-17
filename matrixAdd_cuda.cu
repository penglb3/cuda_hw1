#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include "getopt.hpp"
#else 
#include <unistd.h> 
#endif

#define MAX(x, y) ((x)>(y)?(x):(y))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define TIME_IT_CUDA( exec ) \
{\
    cudaEvent_t start, stop;\
    HANDLE_ERROR( cudaEventCreate(&start) );\
    HANDLE_ERROR( cudaEventCreate(&stop) );\
    cudaEventRecord(start);\
    exec;\
    cudaEventRecord(stop);\
    cudaEventSynchronize(stop);\
    float milliseconds = 0;\
    HANDLE_ERROR( cudaEventElapsedTime(&milliseconds, start, stop) );\
    printf("Time Elapsed: %.3f(ms)\n", milliseconds);\
}

void verify_add(const float* h_A, const float* h_B, const float* h_C, const int numElements){
    float max_error = 0;
    #ifndef _MSC_VER /* MSVC only supports OpenMP 2.0, which does not have max reduction!*/
    #pragma omp parallel for firstprivate(max_error) reduction(max : max_error)
    #endif 
    for (int i = 0; i < numElements; ++i) {
        max_error = MAX(max_error, fabs(h_A[i] + h_B[i] - h_C[i]));
        if (max_error > 1e-5) {
            fprintf(stderr, "Result verification FAILED at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED. Max Error = %.3g\n", max_error);
}
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements){
        C[i] = A[i] + B[i];
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
 __global__ void vectorAddN(const float *A, const float *B, float *C, int numElements, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (numElements+N-1)/N){
        for (int k=0; k<N; k++)
            C[i*N+k] = A[i*N+k] + B[i*N+k];
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the 2D matrix addition of A and B into C. The 3 matrix have the same
 * number of rows (n_rows) and same number of columns (n_cols).
 */
__global__ void matrixAdd2D(float **A, float **B, float **C, int n_rows, int n_cols){
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row index
    int j = blockDim.x * blockIdx.x + threadIdx.x; // column index

    if (i < n_rows && j < n_cols){
        C[i][j] = A[i][j] + B[i][j];
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the 1D matrix addition of A and B into C. The 3 matrix have the same
 * number of rows (n_rows) and same number of columns (n_cols).
 */
__global__ void matrixAdd1D(const float *A, const float *B, float *C, int n_rows, int n_cols){
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row index
    int j = blockDim.x * blockIdx.x + threadIdx.x; // column index
    int k = i*n_cols+j;
    if ((i < n_rows) && (j < n_cols)){
        C[k] = A[k] + B[k];
    }
}
/**
 * CUDA Kernel Device code
 *
 * Initializes the 2D matrix Matrix_2D as Matrix_data. The 2 matrix must have the same
 * number of rows (n_rows) and same number of columns (n_cols).
 */
__global__ void matrixInit2D(float** Matrix_2D, float* Matrix_data, int n_rows, int n_cols){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n_rows)
        Matrix_2D[i] = Matrix_data + i*n_cols;
}

/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
    // Basic information about CUDA driver and runtime environment.
    int version;
    cudaDriverGetVersion(&version);
    printf("===== (18340136) PLB's program.  =====\n");
    printf("===== CUDA Driver Version %d.%d  =====\n", version/1000, version%1000/10);
    cudaRuntimeGetVersion(&version);
    printf("===== CUDA Runtime Version %d.%d =====\n", version/1000, version%1000/10);

    // Program parameters
    int n_rows = 8320, n_cols = 5000, blockDim1D = 256, 
        blockDim2D_x = 16, blockDim2D_y = 16, N = 1;

    // Get parameters from arguments (if provided)
    char c;
    while((c = getopt(argc, argv, "b:r:c:x:y:N:"))!=-1)
        switch(c){
            case 'b': blockDim1D = atoi(optarg);break;
            case 'x': blockDim2D_x = atoi(optarg);break;
            case 'y': blockDim2D_y = atoi(optarg);break;
            case 'r': n_rows = atoi(optarg);break;
            case 'c': n_cols = atoi(optarg);break;
            case 'N': N = atoi(optarg);break;
            default : abort();
        }

    // Print the vector length to be used, and compute its n_bytes
    size_t numElements = n_rows * n_cols;
    size_t n_bytes = numElements * sizeof(float);
    printf("Matrix addition of size %d*%d (=%zd) elements.\n", n_rows, n_cols, numElements);

    /**************************************************************************************
    *  Memory allocation and initialzation.
    **************************************************************************************/
    // Allocate the host input vector A
    float *h_A = (float *)malloc(n_bytes);
    // Allocate the host input vector B
    float *h_B = (float *)malloc(n_bytes);
    // Allocate the host output vector C
    float *h_C = (float *)malloc(n_bytes);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    #pragma omp parallel for
    for (int i = 0; i < numElements; ++i){
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Device vectors.
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    // Allocate the device input vector A
    HANDLE_ERROR( cudaMalloc((void **)&d_A, n_bytes) );
    // Allocate the device input vector B
    HANDLE_ERROR( cudaMalloc((void **)&d_B, n_bytes) );
    // Allocate the device output vector C
    HANDLE_ERROR( cudaMalloc((void **)&d_C, n_bytes) );

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    HANDLE_ERROR( cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice) );

    /**************************************************************************************
    *  2D cuda memory index, 2D memory structure.
    **************************************************************************************/
    // 2D matrix initialization
    float** d_A_2D = NULL, **d_B_2D = NULL, **d_C_2D = NULL;
    HANDLE_ERROR( cudaMalloc((void **)&d_A_2D, sizeof(float*)*n_rows) );
    HANDLE_ERROR( cudaMalloc((void **)&d_B_2D, sizeof(float*)*n_rows) );
    HANDLE_ERROR( cudaMalloc((void **)&d_C_2D, sizeof(float*)*n_rows) );
    int nGrids = (n_rows+blockDim1D-1)/blockDim1D;
    matrixInit2D <<<nGrids, blockDim1D>>> (d_A_2D, d_A, n_rows, n_cols);
    matrixInit2D <<<nGrids, blockDim1D>>> (d_B_2D, d_B, n_rows, n_cols);
    matrixInit2D <<<nGrids, blockDim1D>>> (d_C_2D, d_C, n_rows, n_cols);

    // Compute dims, and print it out.
    dim3 threadsPerBlock2D(blockDim2D_x, blockDim2D_y, 1);
    dim3 blocksPerGrid2D((n_cols+blockDim2D_x-1)/blockDim2D_x, (n_rows+blockDim2D_y-1)/blockDim2D_y , 1);
    printf("\n2D thread index, 2D memory structure: <<< (%d,%d), (%d,%d) >>>\n",
        blocksPerGrid2D.x, blocksPerGrid2D.y, threadsPerBlock2D.x, threadsPerBlock2D.y);

    // Call and time it.
    TIME_IT_CUDA( (matrixAdd2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_A_2D, d_B_2D, d_C_2D, n_rows, n_cols)) );
    HANDLE_ERROR( cudaGetLastError() );

    // Verify results, and clean up device result vector for next test.
    HANDLE_ERROR( cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost) );
    verify_add(h_A, h_B, h_C, numElements);
    HANDLE_ERROR( cudaMemset(d_C, 0, n_bytes) );

    /**************************************************************************************
    *  2D cuda memory index, 1D memory structure.
    **************************************************************************************/
    // 2D-dims already defined. Just use previous results. (and print it.)
    printf("\n2D thread index, 1D memory structure: <<< (%d,%d), (%d,%d) >>>\n",
        blocksPerGrid2D.x, blocksPerGrid2D.y, threadsPerBlock2D.x, threadsPerBlock2D.y);

    // Call and time it.
    TIME_IT_CUDA( (matrixAdd1D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_A, d_B, d_C, n_rows, n_cols)) );
    HANDLE_ERROR( cudaGetLastError() );

    // Verify results, and clean up device result vector for next test.
    HANDLE_ERROR( cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost) );
    verify_add(h_A, h_B, h_C, numElements);
    HANDLE_ERROR( cudaMemset(d_C, 0, n_bytes) );

    if (N > 1){
        /**************************************************************************************
        *  1D cuda memory index, 1D memory structure, N elements per thread. 
        **************************************************************************************/
        // Compute dims, and print it out.
        int threadsPerBlock = blockDim1D, numThreads = (numElements + N - 1) / N;
        int blocksPerGrid =(numThreads + threadsPerBlock - 1) / threadsPerBlock;
        printf("\n1D thread index, 1D memory structure w/ %d elements per thread: <<< %d, %d >>>\n",
            N, blocksPerGrid, threadsPerBlock);

        // Call and time it.
        TIME_IT_CUDA( (vectorAddN<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements, N)) );
        HANDLE_ERROR( cudaGetLastError() );
    } else {
        /**************************************************************************************
        *  1D cuda memory index, 1D memory structure.
        **************************************************************************************/
        // Compute dims, and print it out.
        int threadsPerBlock = blockDim1D;
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
        printf("\n1D thread index, 1D memory structure: <<< %d, %d >>>\n", blocksPerGrid, threadsPerBlock);

        // Call and time it.
        TIME_IT_CUDA( (vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements)) );
        HANDLE_ERROR( cudaGetLastError() );
    }
    // Verify results
    HANDLE_ERROR( cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost) );
    verify_add(h_A, h_B, h_C, numElements);

    /**************************************************************************************
    *  Clean up
    **************************************************************************************/
    // Free device global memory
    HANDLE_ERROR( cudaFree(d_A_2D) );
    HANDLE_ERROR( cudaFree(d_B_2D) );
    HANDLE_ERROR( cudaFree(d_C_2D) );
    HANDLE_ERROR( cudaFree(d_A) );
    HANDLE_ERROR( cudaFree(d_B) );
    HANDLE_ERROR( cudaFree(d_C) );

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    //system("pause");
    return 0;
}