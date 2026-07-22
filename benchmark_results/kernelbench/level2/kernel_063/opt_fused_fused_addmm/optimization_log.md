# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_relu_div_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7750000 ns — **Best latency**: 7750000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_relu_div_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_063/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7750000 ns
- **SM Throughput**: 42.38 % | **Memory**: 57.77 % | **DRAM**: 0.95 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.79 inst/cycle | **No Eligible**: 55.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_register_pressure_relief | 0.950× | 1.001× | ✓ |  |
| C | add_cache_eviction_policies_for_streaming_loads_stores | 0.999× | 1.000× | ✓ |  |
| D | fix_weight_pointer_layout_and_reduce_register_pressure | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_re... | Δ | add_cache_evictio... | Δ | fix_weight_pointe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7750000 ns | 7740000 ns | -10000 ns (-0.1%) | 7750000 ns | +0 ns (+0.0%) | 7750000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 42.38 % | 42.82 % | +0.44 pp | 42.47 % | +0.09 pp | 42.45 % | +0.07 pp |
| Memory Throughput (%) | 57.77 % | 61.73 % | +3.96 pp | 57.89 % | +0.12 pp | 57.87 % | +0.10 pp |
| DRAM Throughput (%) | 0.95 % | 1.04 % | +0.09 pp | 0.95 % | +0.00 pp | 0.95 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.14 % | 54.52 % | -0.62 pp | 55.12 % | -0.02 pp | 55.09 % | -0.05 pp |
| IPC Active | 1.79 inst/cycle | 1.82 inst/cycle | +0.03 | 1.80 inst/cycle | +0.01 | 1.80 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | tf32_tensor_cores_with_register_cap | 0.793× | 0.167× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7750000 ns | 46410000 ns | +38660000 ns (+498.8%) |
| SM Throughput (%) | 42.38 % | 20.46 % | -21.92 pp |
| Memory Throughput (%) | 57.77 % | 59.66 % | +1.89 pp |
| DRAM Throughput (%) | 0.95 % | 38.11 % | +37.16 pp |
| Achieved Occupancy (%) | 12.50 % | 16.63 % | +4.13 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 |
| Waves/SM | 1.94 | 1.29 | -0.65 |
| No Eligible (%) | 55.14 % | 86.08 % | +30.94 pp |
| IPC Active | 1.79 inst/cycle | 0.56 inst/cycle | -1.23 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_tma_block_ptr_loads_to_fix_l1tex_stall_regression | 0.772× | 0.165× | ✓ |  |
| B | tma_loads_via_make_block_ptr | 0.771× | 0.165× | ✓ |  |
| C | restore_tma_block_ptr_loads_to_fix_l1tex_stalls | 0.767× | 0.160× | ✓ |  |
| D | add_triton_autotune | 0.827× | 0.167× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restore_tma_block... | Δ | tma_loads_via_mak... | Δ | restore_tma_block... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7750000 ns | 47000000 ns | +39250000 ns (+506.5%) | 47110000 ns | +39360000 ns (+507.9%) | 48510000 ns | +40760000 ns (+525.9%) | 46380000 ns | +38630000 ns (+498.5%) |
| SM Throughput (%) | 42.38 % | 21.04 % | -21.34 pp | 20.96 % | -21.42 pp | 20.40 % | -21.98 pp | 20.45 % | -21.93 pp |
| Memory Throughput (%) | 57.77 % | 61.09 % | +3.32 pp | 60.94 % | +3.17 pp | 59.18 % | +1.41 pp | 59.72 % | +1.95 pp |
| DRAM Throughput (%) | 0.95 % | 39.87 % | +38.92 pp | 39.80 % | +38.85 pp | 38.70 % | +37.75 pp | 38.46 % | +37.51 pp |
| Achieved Occupancy (%) | 12.50 % | 16.28 % | +3.78 pp | 16.29 % | +3.79 pp | 16.69 % | +4.19 pp | 16.80 % | +4.30 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 128.00 register/thread | -127 | 128.00 register/thread | -127 | 128.00 register/thread | -127 |
| Waves/SM | 1.94 | 1.29 | -0.65 | 1.29 | -0.65 | 1.29 | -0.65 | 1.29 | -0.65 |
| No Eligible (%) | 55.14 % | 86.32 % | +31.18 pp | 86.34 % | +31.20 pp | 86.11 % | +30.97 pp | 85.93 % | +30.79 pp |
| IPC Active | 1.79 inst/cycle | 0.55 inst/cycle | -1.24 | 0.55 inst/cycle | -1.24 | 0.56 inst/cycle | -1.23 | 0.56 inst/cycle | -1.23 |


### Key Learnings

#### ❌ Plan B: `tma_loads_via_make_block_ptr` (0.165×)

Replacing manual pointer arithmetic with tl.make_block_ptr + tl.advance produced no improvement and a slight regression (46.41ms → 47.11ms, speedup 0.1645x). All key metrics remained essentially identical before and after: eligible warps stayed at 0.15, warp cycles per issued instruction stayed at ~19, L1TEX hit rate stayed near 0%, DRAM throughput stayed at ~38-40%, and achieved occupancy barely changed (16.63% → 16.29%). The profiling data indicates the kernel was already in a state where the block_ptr change could not help — the bottleneck (shared memory bank conflicts at 5.6-way, 60% of wavefronts; uncoalesced shared accesses at 56% excessive wavefronts; 128 registers/thread limiting occupancy to 18.75%) was not addressed by the load path change. The TMA block_ptr optimization only helps when the kernel is bottlenecked on global memory address generation and L1TEX latency from strided loads; when shared memory conflicts and register pressure dominate, this change is ineffective.

> **Applicable when**: Never in this kernel state — the TMA block_ptr swap is ineffective when the kernel is already bottlenecked by shared memory bank conflicts (5.6-way average, 60% of wavefronts), uncoalesced shared accesses (56% excessive wavefronts), and register pressure limiting occupancy to 18.75%. The load path change cannot address these bottlenecks.

> **Avoid when**: Do not apply tl.make_block_ptr as a fix when NCU reports SharedMemoryConflicts and UncoalescedSharedAccess as the dominant bottlenecks (speedup_pct 35-46%), and when eligible warps per scheduler remain at 0.15 regardless of load path. The global memory access pattern is not the binding constraint in this regime — shared memory layout and register pressure are.

#### ❌ Plan C: `restore_tma_block_ptr_loads_to_fix_l1tex_stalls` (0.160×)

Replacing manual pointer arithmetic with tl.make_block_ptr/tl.advance (TMA loads) produced no measurable improvement: duration increased from 46.41ms to 48.51ms (0.16x speedup). All key metrics remained essentially unchanged — no-eligible-warp stayed at ~86%, warp cycles per issued instruction stayed at ~19, DRAM throughput stayed at ~38%, and L1TEX hit rate actually dropped slightly from 0.41% to 0.31%. The TMA block_ptr change did not recover the expected 6x speedup documented in memory item a3f2c891, suggesting the kernel's current configuration (128 registers/thread, 65KB dynamic shared memory, 18.75% theoretical occupancy) prevents TMA from providing its prefetching benefit, or the kernel is running in a different code path than expected.

> **Applicable when**: Never in this configuration — switching between manual pointer arithmetic and TMA block_ptr does not resolve the underlying bottleneck when the kernel is already register-limited (128 regs/thread), shared-memory-limited (65KB/block), and occupancy-limited (18.75% theoretical). The root cause of the 86% no-eligible-warp stall is not the load mechanism alone.

> **Avoid when**: Do not apply tl.make_block_ptr restoration as a standalone fix when the kernel already exhibits 86% no-eligible-warp, 19 warp cycles/instruction, and 18.75% theoretical occupancy — these metrics indicate the bottleneck is structural (occupancy, register pressure, shared memory) and not purely the memory access pattern. Switching load mechanisms alone will not recover performance in this regime.

---

## Cumulative Progress

```
Baseline:  7750000 ns  (1.000×)

Final best: 7750000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,16 +2,27 @@

 import triton
 import triton.language as tl
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=2, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=2, maxnreg=128),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=2, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=2, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=5, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=4, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8, maxnreg=128),
     ],
     key=['M', 'N', 'K'],
 )
@@ -44,7 +55,8 @@

     # Pointers
     X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
     # W is (N, K) -- weight matrix, we compute X @ W.T so we load W[n, k]
-    W_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
+    # K is row-index (axis 0), N is column-index (axis 1) for coalesced N access
+    W_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
@@ -78,7 +90,6 @@

     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     Out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
     tl.store(Out_ptrs, acc.to(tl.float32), mask=out_mask)
-
 
 def kernel_function(weight, bias, x):
     # weight: (N, K) = (8192, 8192)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
