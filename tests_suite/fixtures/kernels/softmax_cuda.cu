#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// =============================================================================
// Warp-level reduction operations
// =============================================================================

template<int WARP_SIZE = 32>
__device__ __forceinline__ float warp_max(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<int WARP_SIZE = 32>
__device__ __forceinline__ float warp_sum(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// =============================================================================
// Block-level reduction operations
// =============================================================================

template<int BLOCK_SIZE = 256>
__device__ __forceinline__ float block_sum(float val) {
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    static __shared__ float shared_mem[NUM_WARPS];

    const float warp_sum_val = warp_sum<WARP_SIZE>(val);

    if (lane_id == 0) {
        shared_mem[warp_id] = warp_sum_val;
    }
    __syncthreads();

    const float thread_val = (lane_id < NUM_WARPS) ? shared_mem[lane_id] : 0.0f;
    return warp_sum<WARP_SIZE>(thread_val);
}

template<int BLOCK_SIZE = 256>
__device__ __forceinline__ float block_max(float val) {
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    static __shared__ float shared_mem[NUM_WARPS];

    const float warp_max_val = warp_max<WARP_SIZE>(val);

    if (lane_id == 0) {
        shared_mem[warp_id] = warp_max_val;
    }
    __syncthreads();

    const float thread_val = (lane_id < NUM_WARPS) ? shared_mem[lane_id] : -FLT_MAX;
    return warp_max<WARP_SIZE>(thread_val);
}

// =============================================================================
// Softmax kernel
// =============================================================================

template<int BLOCK_SIZE = 256>
__global__ void softmax_kernel(const float* input, float* output, const int N) {
    const int tid = threadIdx.x;

    // Step 1: max
    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, input[i]);
    max_val = block_max<BLOCK_SIZE>(max_val);

    // Step 2: sum of exp
    float sum_val = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        sum_val += expf(input[i] - max_val);
    sum_val = block_sum<BLOCK_SIZE>(sum_val);

    // Step 3: normalize
    for (int i = tid; i < N; i += blockDim.x)
        output[i] = expf(input[i] - max_val) / sum_val;
}

extern "C" void solve(const float* input, float* output, const int N) {
    constexpr int THREADS_PER_BLOCK = 1024;
    softmax_kernel<THREADS_PER_BLOCK><<<1, THREADS_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}

// =============================================================================
// Host driver
// =============================================================================

static void fill_random(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
}

static int check_correctness(int N) {
    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *d_in, *d_out;

    srand(42);
    fill_random(h_in, N);

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference
    float max_v = -FLT_MAX;
    for (int i = 0; i < N; i++) max_v = fmaxf(max_v, h_in[i]);
    float sum_v = 0.0f;
    for (int i = 0; i < N; i++) sum_v += expf(h_in[i] - max_v);

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float ref = expf(h_in[i] - max_v) / sum_v;
        if (fabsf(ref - h_out[i]) > 1e-4f) {
            printf("FAIL at i=%d: ref=%.6f got=%.6f\n", i, ref, h_out[i]);
            ok = 0; break;
        }
    }
    if (ok) printf("Correctness: PASS\n");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return ok;
}

int main(int argc, char** argv) {
    int N = 4096;
    int profile_only = 0;
    int benchmark = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--profile-only") == 0)
            profile_only = 1;
        else if (strcmp(argv[i], "--benchmark") == 0)
            benchmark = 1;
    }

    printf("softmax_cuda  N=%d\n", N);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    float *h_in = (float*)malloc(N * sizeof(float));
    srand(0);
    fill_random(h_in, N);
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_in);

    if (profile_only) {
        solve(d_in, d_out, N);
        printf("Profile-only run complete.\n");
    } else if (benchmark) {
        // Warmup
        for (int i = 0; i < 5; i++) solve(d_in, d_out, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int REPS = 200;
        cudaEventRecord(start);
        for (int i = 0; i < REPS; i++) solve(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Avg latency: %.3f us\n", ms / REPS * 1000.0f);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        check_correctness(N);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
