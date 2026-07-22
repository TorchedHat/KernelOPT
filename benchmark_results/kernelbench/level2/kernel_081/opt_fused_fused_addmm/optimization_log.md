# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_swish_epilogue_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8140000 ns — **Best latency**: 8150000 ns (**0.9988×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_swish_epilogue_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_081/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 8140000 ns
- **SM Throughput**: 42.53 % | **Memory**: 58.30 % | **DRAM**: 0.97 %
- **Achieved Occupancy**: 12.49 %
- **IPC Active**: 1.82 inst/cycle | **No Eligible**: 54.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_smaller_tiles_and_num_warps | 0.950× | 0.318× | ✓ |  |
| C | eviction_policies_streaming_ab_tiles | 0.999× | 0.998× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles_and_more_stages | 0.607× | 0.334× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8140000 ns | 25600000 ns | +17460000 ns (+214.5%) | 8160000 ns | +20000 ns (+0.2%) | 24340000 ns | +16200000 ns (+199.0%) |
| SM Throughput (%) | 42.53 % | 13.84 % | -28.69 pp | 42.54 % | +0.01 pp | 14.41 % | -28.12 pp |
| Memory Throughput (%) | 58.30 % | 96.13 % | +37.83 pp | 58.31 % | +0.01 pp | 95.30 % | +37.00 pp |
| DRAM Throughput (%) | 0.97 % | 1.04 % | +0.07 pp | 0.96 % | -0.01 pp | 1.02 % | +0.05 pp |
| Achieved Occupancy (%) | 12.49 % | 27.28 % | +14.79 pp | 12.49 % | +0.00 pp | 24.49 % | +12.00 pp |
| Registers/Thread | 255.00 register/thread | 96.00 register/thread | -159 | 255.00 register/thread | +0 | 96.00 register/thread | -159 |
| Waves/SM | 1.94 | 6.90 | +4.96 | 1.94 | +0.00 | 7.76 | +5.82 |
| No Eligible (%) | 54.54 % | 85.92 % | +31.38 pp | 54.55 % | +0.01 pp | 85.14 % | +30.60 pp |
| IPC Active | 1.82 inst/cycle | 0.56 inst/cycle | -1.26 | 1.82 inst/cycle | +0.00 | 0.59 inst/cycle | -1.23 |


### Key Learnings

#### ❌ Plan B: `reduce_register_pressure_via_smaller_tiles_and_num_warps` (0.318×)

Reducing tile sizes (BLOCK_M/BLOCK_N from 128 to 32-64) and num_warps to lower register pressure successfully raised theoretical occupancy from 12.5% to 28.1% and reduced registers/thread from 255 to 96. However, this caused a severe regression: Warp Cycles Per Issued Instruction jumped from 4.39 to 31.06 cycles, and shared memory bank conflicts worsened dramatically from 5.4-way (58.51% of wavefronts) to 18-way (83.34% of wavefronts), with uncoalesced shared wavefronts rising from 56% to 76%. The smaller tiles increased the number of grid blocks (512→8192) and waves (1.94→6.9), but the MIO pipeline became the dominant bottleneck: 42.6% of stall cycles were scoreboard waits on MIO and 37.3% were MIO queue-full stalls, making the kernel 3.1x slower despite higher occupancy.

> **Applicable when**: Never: this specific direction (shrinking tiles to reduce register pressure) is counterproductive when the kernel has a complex epilogue using shared memory with non-coalesced access patterns. The bank conflict penalty from smaller tiles overwhelms any occupancy benefit.

> **Avoid when**: When the kernel already has significant shared memory bank conflicts (>50% of wavefronts uncoalesced) — shrinking tile sizes increases the number of blocks and shared memory accesses per output element, amplifying bank conflicts. When the kernel uses libdevice special math functions (sigmoid, tanh) that stress the MIO pipeline, smaller tiles increase MIO pressure per unit of output. When register pressure is the bottleneck but shared memory access patterns are non-coalesced, fix the access pattern first before reducing tile sizes.

#### ❌ Plan D: `reduce_register_pressure_via_smaller_tiles_and_more_stages` (0.334×)

Reducing BLOCK_M/BLOCK_N tile sizes to lower register pressure successfully cut registers/thread (255→96) and improved theoretical occupancy (12.5%→25%), but caused a 3x regression. The smaller tiles dramatically worsened shared memory bank conflicts (5.4-way→17-way average) and uncoalesced shared wavefronts (56%→81%), causing warp cycles per issued instruction to jump from 4.39 to 26.37 and eligible warps per scheduler to drop from 0.76 to 0.18. The net effect was that the L1/TEX cache became the bottleneck at 98.28% throughput and compute throughput collapsed from 42.53% to 14.41%, making the kernel 3x slower despite better occupancy.

> **Applicable when**: Never — this specific direction (smaller tiles + more stages to reduce register pressure) caused severe regression in this kernel due to worsened shared memory bank conflicts.

> **Avoid when**: When the kernel has significant shared memory bank conflicts even before the change (5.4-way conflicts in the baseline). Reducing tile sizes in Triton GEMM kernels with complex epilogues can dramatically worsen shared memory access patterns, increasing bank conflicts from 5x to 17x and causing L1/TEX saturation (98%+). The occupancy gain does not compensate when eligible warps per scheduler actually decreases (0.76→0.18) due to MIO stalls from bank conflicts. Do not apply when: (1) baseline already has shared memory bank conflicts >4-way, (2) the kernel uses shared memory for GEMM tiles where smaller tile dimensions create worse stride patterns, (3) the bottleneck is shared memory access pattern rather than pure register pressure.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | epilogue_fp16_cast_to_reduce_register_spill | 0.998× | 0.999× | ✓ |  |
| B | restore_large_tiles_to_eliminate_shared_memory_bank_conflicts | 1.031× | 1.012× | ✓ |  |
| C | restore_large_tiles_to_eliminate_shared_memory_bank_conflicts | 1.032× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | epilogue_fp16_cas... | Δ | restore_large_til... | Δ | restore_large_til... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8140000 ns | 8150000 ns | +10000 ns (+0.1%) | 8040000 ns | -100000 ns (-1.2%) | 8150000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 42.53 % | 42.46 % | -0.07 pp | 42.80 % | +0.27 pp | 42.62 % | +0.09 pp |
| Memory Throughput (%) | 58.30 % | 58.18 % | -0.12 pp | 57.99 % | -0.31 pp | 58.43 % | +0.13 pp |
| DRAM Throughput (%) | 0.97 % | 0.97 % | +0.00 pp | 0.96 % | -0.01 pp | 1.00 % | +0.03 pp |
| Achieved Occupancy (%) | 12.49 % | 12.49 % | +0.00 pp | 12.50 % | +0.01 pp | 12.49 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.54 % | 54.43 % | -0.11 pp | 54.85 % | +0.31 pp | 54.46 % | -0.08 pp |
| IPC Active | 1.82 inst/cycle | 1.82 inst/cycle | +0.00 | 1.80 inst/cycle | -0.02 | 1.82 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | simplify_epilogue_to_reduce_register_pressure_and_spill | 1.014× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | simplify_epilogue... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8140000 ns | 8100000 ns | -40000 ns (-0.5%) |
| SM Throughput (%) | 42.53 % | 42.54 % | +0.01 pp |
| Memory Throughput (%) | 58.30 % | 58.32 % | +0.02 pp |
| DRAM Throughput (%) | 0.97 % | 0.97 % | +0.00 pp |
| Achieved Occupancy (%) | 12.49 % | 12.50 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 54.54 % | 54.14 % | -0.40 pp |
| IPC Active | 1.82 inst/cycle | 1.83 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_redundant_epilogue_clamp_to_reduce_register_spill | 1.016× | 1.010× | ✓ |  |
| D | grid_alignment_and_larger_block_k_configs | 1.015× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | grid_alignment_an... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8140000 ns | 8060000 ns | -80000 ns (-1.0%) | 8100000 ns | -40000 ns (-0.5%) |
| SM Throughput (%) | 42.53 % | 42.84 % | +0.31 pp | 42.91 % | +0.38 pp |
| Memory Throughput (%) | 58.30 % | 58.72 % | +0.42 pp | 58.85 % | +0.55 pp |
| DRAM Throughput (%) | 0.97 % | 0.98 % | +0.01 pp | 0.97 % | +0.00 pp |
| Achieved Occupancy (%) | 12.49 % | 12.49 % | +0.00 pp | 12.50 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.54 % | 54.25 % | -0.29 pp | 54.15 % | -0.39 pp |
| IPC Active | 1.82 inst/cycle | 1.84 inst/cycle | +0.02 | 1.83 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  8140000 ns  (1.000×)

Final best: 8150000 ns  (0.9988×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 
 @triton.autotune(
@@ -52,43 +53,42 @@

 
         a = tl.load(a_ptrs,
                     mask=(offs_m[:, None] < M) & (mask_k[None, :]),
-                    other=0.0)
+                    other=0.0,
+                    eviction_policy='evict_first')
         b = tl.load(b_ptrs,
                     mask=(mask_k[:, None]) & (offs_n[None, :] < N),
-                    other=0.0)
+                    other=0.0,
+                    eviction_policy='evict_first')
 
         acc = tl.dot(a, b, acc, allow_tf32=False)
 
         a_ptrs += BLOCK_K * stride_ak
         b_ptrs += BLOCK_K * stride_bk
 
-    # Load bias and add
+    # Load bias and add in FP32 (preserve precision for large bias values)
     bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
     acc = acc + bias[None, :]
 
-    # Swish: x * sigmoid(x)
+    # Swish: x * sigmoid(x) — keep in FP32 for precision near 0
     sigmoid_acc = tl.sigmoid(acc)
     acc = acc * sigmoid_acc
 
     # Divide by 2
     acc = acc * 0.5
 
-    # Clamp to [-1, 1]
+    # Clamp to [-1, 1] in FP32 — necessary because swish(x)/2 can exceed [-1,1]
+    # for large |x| (swish(x)/2 → x/2 as x → +inf)
     acc = tl.maximum(acc, -1.0)
     acc = tl.minimum(acc, 1.0)
 
-    # Tanh
+    # Tanh directly in FP32 — eliminates the FP16 cast round-trip and the
+    # redundant outer clamp (tanh output is always in (-1,1) by definition)
     acc = libdevice.tanh(acc)
 
-    # Clamp to [-1, 1] again
-    acc = tl.maximum(acc, -1.0)
-    acc = tl.minimum(acc, 1.0)
-
-    # Store result
+    # Store result directly as FP32 — no outer clamp needed (tanh is bounded)
     c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
     mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
-    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)
-
+    tl.store(c_ptrs, acc, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(p_gemm_weight, p_gemm_bias, x):
     # x: [M, K], weight: [N, K] (linear weight is transposed), bias: [N]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
