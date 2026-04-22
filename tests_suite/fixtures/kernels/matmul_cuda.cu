// tests_suite/fixtures/kernels/matmul_cuda.cu
//
// Tiled matrix multiplication: C = A x B (fp32, square N x N).
// Kernel only — no main(), no correctness check.
// Correctness is verified externally via the agentic checker.

#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

// Entry point for NCU profiling (called from host benchmarking code).
extern "C" void matmul(const float* A, const float* B, float* C, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
