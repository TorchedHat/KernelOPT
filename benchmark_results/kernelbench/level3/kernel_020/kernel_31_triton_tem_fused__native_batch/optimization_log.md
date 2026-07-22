# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 36770 ns — **Best latency**: 36510 ns (**1.0071×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tma_loads_hopper_tensor_memory_accelerator`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_31_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (90,1,1), Waves/SM: 0.23
- **Registers/thread**: 168
- **Duration**: 36770 ns
- **SM Throughput**: 17.74 % | **Memory**: 39.83 % | **DRAM**: 0.79 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 70.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.994× | 1.003× | ✓ |  |
| B | tma_loads_hopper_tensor_memory_accelerator | 1.030× | 1.007× | ✓ | **✓ best** |
| C | eviction_policy_streaming_b_matrix | 0.994× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_hopper_... | Δ | eviction_policy_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 36770 ns | 36670 ns | -100 ns (-0.3%) | 36510 ns | -260 ns (-0.7%) | 36770 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 17.74 % | 17.67 % | -0.07 pp | 17.57 % | -0.17 pp | 17.65 % | -0.09 pp |
| Memory Throughput (%) | 39.83 % | 39.68 % | -0.15 pp | 39.81 % | -0.02 pp | 39.61 % | -0.22 pp |
| DRAM Throughput (%) | 0.79 % | 0.79 % | +0.00 pp | 0.79 % | +0.00 pp | 0.79 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 70.69 % | 70.58 % | -0.11 pp | 70.84 % | +0.15 pp | 70.59 % | -0.10 pp |
| IPC Active | 1.18 inst/cycle | 1.17 inst/cycle | -0.01 | 1.17 inst/cycle | -0.01 | 1.16 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0071× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_b_matrix_order | 1.025× | 1.004× | ✓ |  |
| B | fix_b_matrix_load_coalescing_and_add_autotune | 0.994× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 36770 ns | 36640 ns | -130 ns (-0.4%) | 38020 ns | +1250 ns (+3.4%) |
| SM Throughput (%) | 17.74 % | 17.63 % | -0.11 pp | 16.92 % | -0.82 pp |
| Memory Throughput (%) | 39.83 % | 39.94 % | +0.11 pp | 38.13 % | -1.70 pp |
| DRAM Throughput (%) | 0.79 % | 0.79 % | +0.00 pp | 0.76 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 |
| No Eligible (%) | 70.69 % | 70.59 % | -0.10 pp | 72.69 % | +2.00 pp |
| IPC Active | 1.18 inst/cycle | 1.18 inst/cycle | +0.00 | 1.09 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.0071× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_block_ptr_order_and_reduce_register_pressure_with_maxnreg | 0.856× | 0.839× | ✓ |  |
| B | cap_registers_with_maxnreg_to_improve_occupancy | 0.855× | 0.842× | ✓ |  |
| C | fix_b_matrix_shared_memory_bank_conflicts_via_block_ptr_transpose | 1.003× | 0.986× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bloc... | Δ | cap_registers_wit... | Δ | fix_b_matrix_shar... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 36770 ns | 43810 ns | +7040 ns (+19.1%) | 43650 ns | +6880 ns (+18.7%) | 37280 ns | +510 ns (+1.4%) |
| SM Throughput (%) | 17.74 % | 16.47 % | -1.27 pp | 16.53 % | -1.21 pp | 16.99 % | -0.75 pp |
| Memory Throughput (%) | 39.83 % | 35.80 % | -4.03 pp | 35.93 % | -3.90 pp | 38.94 % | -0.89 pp |
| DRAM Throughput (%) | 0.79 % | 0.66 % | -0.13 pp | 0.67 % | -0.12 pp | 0.78 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 128.00 register/thread | -40 | 128.00 register/thread | -40 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.17 | -0.06 | 0.17 | -0.06 | 0.23 | +0.00 |
| No Eligible (%) | 70.69 % | 72.29 % | +1.60 pp | 72.30 % | +1.61 pp | 72.80 % | +2.11 pp |
| IPC Active | 1.18 inst/cycle | 1.11 inst/cycle | -0.07 | 1.11 inst/cycle | -0.07 | 1.09 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan B: `cap_registers_with_maxnreg_to_improve_occupancy` (0.842×)

Adding maxnreg=128 to Triton configs successfully reduced registers/thread from 168 to 128 and improved Block Limit Registers from 3 to 4, raising theoretical occupancy from 18.75% to 25%. However, achieved occupancy remained unchanged at 6.25% and kernel duration increased from 36.64 µs to 43.65 µs (0.84x regression). The register cap forced the compiler to spill registers to local memory: new MemoryCacheAccessPattern rules appeared flagging uncoalesced local loads and stores (only 1 of 32 bytes per sector utilized), and L2 Compression Input Sectors ballooned from 93K to 422K. The instruction count also increased from 5.1M to 5.7M, indicating spill-induced overhead. When a kernel is already occupancy-limited by grid size (90 blocks < 132 SMs, 0.23 waves), capping registers does not help if the achieved occupancy is already pinned by the wave count — and the resulting register spills add net overhead.

> **Applicable when**: Never beneficial in this configuration. Theoretically applicable when: register pressure is the sole occupancy limiter, the grid is large enough to benefit from higher occupancy (waves_per_sm >= 1.0), and the kernel has sufficient arithmetic intensity to absorb spill latency.

> **Avoid when**: Do NOT apply maxnreg capping when: (1) the grid is too small to fill the GPU (waves_per_sm < 0.5, grid < SM count) — higher theoretical occupancy cannot be realized; (2) the kernel is already near its register budget boundary where any cap will cause spilling to local memory (MemoryCacheAccessPattern rules appear post-cap); (3) achieved occupancy remains unchanged after the cap, indicating the bottleneck is elsewhere (grid size, shared memory, or wave count). In this kernel, 90 blocks across 132 SMs means even perfect per-SM occupancy cannot help — the wave count is the binding constraint.

---

## Cumulative Progress

```
Baseline:  36770 ns  (1.000×)
Iter 1     36510 ns  (1.007×)  — tma_loads_hopper_tensor_memory_accelerator

Final best: 36510 ns  (1.0071×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,37 +56,32 @@

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
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create block pointers for TMA loads (eliminates scalar pointer arithmetic and bank conflicts)
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 192*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1152*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +93,6 @@

     # inductor generates a suffix
     xindex = idx_n + 1152*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
