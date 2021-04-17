#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <ctime>
#include <chrono>
#ifdef _WIN32
#include "getopt.hpp"
#else 
#include <unistd.h> 
#endif

#define MAX(x, y) ((x)>(y)?(x):(y))

typedef std::chrono::high_resolution_clock Clock;
auto t1 = Clock::now(), t2 = Clock::now();

#define TIME_IT_START t1 = Clock::now();

#define TIME_IT_STOP \
{\
    t2 = Clock::now();\
    printf("Time Elapsed: %f(ms)\n", (t2 - t1)/100000.0);\
}

void verify_add(const float* h_A, const float* h_B, const float* h_C, const int numElements){
    float max_error = 0;
    #pragma omp parallel for
    for (int i = 0; i < numElements; ++i) {
        max_error = MAX(max_error, fabs(h_A[i] + h_B[i] - h_C[i]));
        if (max_error > 1e-5) {
            fprintf(stderr, "Result verification FAILED at element %d!\n", i);
            system("pause");
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED. Max Error = %.3g\n", max_error);
}

int main(int argc, char* argv[]){
    char c;
    int n_rows = 8320, n_cols = 5000, num_threads = 8;
    while((c = getopt(argc, argv, "r:c:t:"))!=-1)
        switch(c){
            case 't': num_threads = atoi(optarg);break;
            case 'r': n_rows = atoi(optarg);break;
            case 'c': n_cols = atoi(optarg);break;
            default : abort();
        }
    size_t numElements = n_rows * n_cols;
    size_t n_bytes = numElements * sizeof(float);
    printf("===== (18340136) PLB's program.  =====\n");
    printf("===== OpenMP Version %d.     =====\n", _OPENMP);
    printf("Matrix addition of size %d*%d (=%d) elements.\n", n_rows, n_cols, numElements);
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

    printf("\nCPU sequential addition\n");
    TIME_IT_START
    for (int i = 0; i < numElements; ++i){
        h_C[i] = h_A[i] + h_B[i];
    }
    TIME_IT_STOP
    verify_add(h_A, h_B, h_C, numElements);
    
    printf("\nCPU parallel addition with %d OpenMP threads\n", num_threads);
    TIME_IT_START
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < numElements; ++i){
        h_C[i] = h_A[i] + h_B[i];
    }
    TIME_IT_STOP
    verify_add(h_A, h_B, h_C, numElements);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}