# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_58` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 30910 ns — **Best latency**: 37790 ns (**0.8179×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_58`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_58_triton_tem_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62,1,1), Waves/SM: 0.06
- **Registers/thread**: 64
- **Duration**: 30910 ns
- **SM Throughput**: 5.76 % | **Memory**: 37.84 % | **DRAM**: 2.70 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.52 inst/cycle | **No Eligible**: 86.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.993× | 1.001× | ✓ |  |
| B | tma_loads_with_larger_tiles_and_autotune | 0.465× | 0.445× | ✓ |  |
| C | memory_coalescing_fix_b_matrix_bank_conflicts_with_autotune | 0.991× | 0.993× | ✓ |  |
| D | autotune_tile_sizes_to_increase_parallelism_and_fix_bank_conflicts | 0.991× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_la... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30910 ns | 30880 ns | -30 ns (-0.1%) | 69410 ns | +38500 ns (+124.6%) | 31140 ns | +230 ns (+0.7%) | 31040 ns | +130 ns (+0.4%) |
| SM Throughput (%) | 5.76 % | 5.78 % | +0.02 pp | 9.06 % | +3.30 pp | 5.78 % | +0.02 pp | 5.76 % | +0.00 pp |
| Memory Throughput (%) | 37.84 % | 37.98 % | +0.14 pp | 38.38 % | +0.54 pp | 37.99 % | +0.15 pp | 37.86 % | +0.02 pp |
| DRAM Throughput (%) | 2.70 % | 2.70 % | +0.00 pp | 1.21 % | -1.49 pp | 2.68 % | -0.02 pp | 2.69 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 168.00 register/thread | +104 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.23 | +0.17 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 86.86 % | 86.86 % | +0.00 pp | 79.74 % | -7.12 pp | 86.85 % | -0.01 pp | 87.01 % | +0.15 pp |
| IPC Active | 0.52 inst/cycle | 0.52 inst/cycle | +0.00 | 0.81 inst/cycle | +0.29 | 0.52 inst/cycle | +0.00 | 0.52 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_shared_memory_transpose_with_padding | 1.000× | 0.993× | ✓ |  |
| B | shared_memory_padding_to_eliminate_b_matrix_bank_conflicts | 0.999× | 0.994× | ✓ |  |
| C | fix_b_matrix_bank_conflicts_via_transposed_load | 0.997× | 0.996× | ✓ |  |
| D | revert_to_scalar_pointers_with_small_tiles_to_restore_occupancy | 1.022× | 0.818× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | shared_memory_pad... | Δ | fix_b_matrix_bank... | Δ | revert_to_scalar_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 30910 ns | 31140 ns | +230 ns (+0.7%) | 31100 ns | +190 ns (+0.6%) | 31040 ns | +130 ns (+0.4%) | 37790 ns | +6880 ns (+22.3%) |
| SM Throughput (%) | 5.76 % | 5.74 % | -0.02 pp | 5.75 % | -0.01 pp | 5.74 % | -0.02 pp | 4.59 % | -1.17 pp |
| Memory Throughput (%) | 37.84 % | 37.73 % | -0.11 pp | 37.79 % | -0.05 pp | 37.71 % | -0.13 pp | 30.50 % | -7.34 pp |
| DRAM Throughput (%) | 2.70 % | 2.68 % | -0.02 pp | 2.68 % | -0.02 pp | 2.69 % | -0.01 pp | 2.21 % | -0.49 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 60.00 register/thread | -4 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.06 | +0.00 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 86.86 % | 86.85 % | -0.01 pp | 86.85 % | -0.01 pp | 86.87 % | +0.01 pp | 89.35 % | +2.49 pp |
| IPC Active | 0.52 inst/cycle | 0.53 inst/cycle | +0.01 | 0.52 inst/cycle | +0.00 | 0.52 inst/cycle | +0.00 | 0.43 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | software_pipeline_num_stages_to_hide_mio_latency | 0.992× | 0.776× | ✓ |  |
| D | increase_num_stages_to_hide_mio_scoreboard_stall_latency | 1.019× | 0.821× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 30910 ns | 39810 ns | +8900 ns (+28.8%) | 37630 ns | +6720 ns (+21.7%) |
| SM Throughput (%) | 5.76 % | 4.22 % | -1.54 pp | 4.61 % | -1.15 pp |
| Memory Throughput (%) | 37.84 % | 30.71 % | -7.13 pp | 30.65 % | -7.19 pp |
| DRAM Throughput (%) | 2.70 % | 2.09 % | -0.61 pp | 2.22 % | -0.48 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 53.00 register/thread | -11 | 60.00 register/thread | -4 |
| Waves/SM | 0.06 | 0.05 | -0.01 | 0.06 | +0.00 |
| No Eligible (%) | 86.86 % | 90.42 % | +3.56 pp | 89.29 % | +2.43 pp |
| IPC Active | 0.52 inst/cycle | 0.38 inst/cycle | -0.14 | 0.43 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan D: `increase_num_stages_to_hide_mio_scoreboard_stall_latency` (0.821×)

Extending the Triton autotune config list to include num_stages in {5,6,7,8} (alongside the original {2,3,4}) caused a 22% regression in kernel duration (31.04us → 37.63us). Despite the intent to hide MIO scoreboard stalls via deeper software pipelining, the winning autotuned config actually reduced dynamic shared memory per block (16KB → 8KB) and registers per thread (64 → 60), indicating the autotuner selected a lower-stage config — yet warp cycles per issued instruction worsened (7.62 → 9.30) and memory throughput dropped (131.81 → 108.71 GB/s). The profiling signals are nearly identical to the iteration-2 masked-load regression, strongly suggesting the code diff's introduction of explicit mask predicates (mask_a, mask_b) in the inner loop — not the num_stages change — is the actual causal regression, with the autotuner selecting a config that happens to use less shared memory. When a kernel is severely latency-bound with only 0.06 waves/SM and 6.25% achieved occupancy, adding more pipeline stages cannot compensate because there are not enough concurrent warps to fill the pipeline, and any accompanying code changes (e.g., masked loads) dominate the performance outcome.

> **Applicable when**: Never: this strategy caused regression. The anti-pattern is expanding the num_stages autotune range while simultaneously introducing explicit mask predicates in the inner loop. Even if the num_stages change alone were neutral, the masked loads dominate and cause regression.

> **Avoid when**: Do not expand num_stages autotune configs when: (1) the kernel is severely latency-bound with waves_per_sm < 0.1 and achieved occupancy ~6%, where pipeline depth cannot be exploited without more concurrent warps; (2) the code change is bundled with other modifications (e.g., masked loads, pointer arithmetic refactoring) that independently cause regression; (3) the dominant bottleneck is shared memory bank conflicts (12-way, 77%+ excessive wavefronts) rather than pure pipeline latency — bank conflicts serialize within each pipeline stage regardless of num_stages. On small grids (grid_size << sm_count), the fundamental occupancy ceiling prevents software pipelining from being effective.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | restore_transposed_b_load_with_restricted_num_stages_autotune | 0.993× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_transpose... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 30910 ns | 31010 ns | +100 ns (+0.3%) |
| SM Throughput (%) | 5.76 % | 5.75 % | -0.01 pp |
| Memory Throughput (%) | 37.84 % | 37.78 % | -0.06 pp |
| DRAM Throughput (%) | 2.70 % | 2.69 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.06 | +0.00 |
| No Eligible (%) | 86.86 % | 86.91 % | +0.05 pp |
| IPC Active | 0.52 inst/cycle | 0.52 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  30910 ns  (1.000×)

Final best: 37790 ns  (0.8179×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,76 +5,75 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original configs
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        # New high-pipeline-depth configs to hide MIO stall
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=6),
+        triton.Config({}, num_warps=8, num_stages=6),
+        triton.Config({}, num_warps=4, num_stages=7),
+        triton.Config({}, num_warps=8, num_stages=7),
+        triton.Config({}, num_warps=4, num_stages=8),
+        triton.Config({}, num_warps=8, num_stages=8),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_58(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
+    ALLOW_TF32: tl.constexpr = False
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 32
+    BLOCK_N: tl.constexpr = 32
+    BLOCK_K: tl.constexpr = 32
+    GROUP_M: tl.constexpr = 8
 
     M = 1960
     N = 32
     K = 512
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 512
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 512
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    pid = tl.program_id(0)
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
 
     # re-order program ID for better L2 performance
     width = GROUP_M * grid_n
     group_id = pid // width
     group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
     pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
+    pid_n = (pid % width) // group_size
+
     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
+    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
     offs_k = tl.arange(0, BLOCK_K)
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+    for k_idx in range(tl.cdiv(K, BLOCK_K)):
+        k_start = k_idx * BLOCK_K
+        # A is row-major with strides (512, 1)
+        a_ptrs = arg_A + offs_m[:, None] * 512 + (k_start + offs_k[None, :]) * 1
+        mask_a = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
+        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
+        # B is column-major with strides (1, 512)
+        b_ptrs = arg_B + (k_start + offs_k[:, None]) * 1 + offs_n[None, :] * 512
+        mask_b = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)
+        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,12 +83,9 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 32*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + idx_n, mask=(idx_n < N), eviction_policy='evict_last')
     tmp1 = acc + tmp0
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+    tl.store(out_ptr0 + (idx_n + N * idx_m), tmp1, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
