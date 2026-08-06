# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3960000 ns — **Best latency**: 1600000 ns (**2.4750×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fast_accumulation_fp16_dot_with_reduced_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_009/synthesis/kernel_0_triton_tem_fused_mm_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (65536,1,1), Waves/SM: 496.48
- **Registers/thread**: 184
- **Duration**: 3960000 ns
- **SM Throughput**: 47.34 % | **Memory**: 45.55 % | **DRAM**: 22.67 %
- **Achieved Occupancy**: 12.41 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 51.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.188× | 1.005× | ✓ |  |
| B | fast_accumulation_fp16_dot_with_reduced_tile_sizes | 1.885× | 2.475× | ✓ | **✓ best** |
| C | fp8_precision_tl_dot_operands | 1.205× | 1.571× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles | 1.083× | 0.507× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fast_accumulation... | Δ | fp8_precision_tl_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3960000 ns | 3940000 ns | -20000 ns (-0.5%) | 1600000 ns | -2360000 ns (-59.6%) | 2520000 ns | -1440000 ns (-36.4%) | 7810000 ns | +3850000 ns (+97.2%) |
| SM Throughput (%) | 47.34 % | 31.67 % | -15.67 pp | 32.30 % | -15.04 pp | 43.30 % | -4.04 pp | 58.46 % | +11.12 pp |
| Memory Throughput (%) | 45.55 % | 69.18 % | +23.63 pp | 69.54 % | +23.99 pp | 79.20 % | +33.65 pp | 94.99 % | +49.44 pp |
| DRAM Throughput (%) | 22.67 % | 22.78 % | +0.11 pp | 55.93 % | +33.26 pp | 35.64 % | +12.97 pp | 11.16 % | -11.51 pp |
| Achieved Occupancy (%) | 12.41 % | 11.91 % | -0.50 pp | 23.64 % | +11.23 pp | 17.74 % | +5.33 pp | 71.18 % | +58.77 pp |
| Registers/Thread | 184.00 register/thread | 96.00 register/thread | -88 | 125.00 register/thread | -59 | 128.00 register/thread | -56 | 40.00 register/thread | -144 |
| Waves/SM | 496.48 | 992.97 | +496.49 | 248.24 | -248.24 | 661.98 | +165.50 | 1323.96 | +827.48 |
| No Eligible (%) | 51.48 % | 68.34 % | +16.86 pp | 67.79 % | +16.31 pp | 58.22 % | +6.74 pp | 41.52 % | -9.96 pp |
| IPC Active | 1.93 inst/cycle | 1.27 inst/cycle | -0.66 | 1.29 inst/cycle | -0.64 | 1.67 inst/cycle | -0.26 | 2.34 inst/cycle | +0.41 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_via_smaller_tiles` (0.507×)

Reducing tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) to lower register pressure and increase occupancy can backfire when the resulting smaller tiles cause L1/TEX cache saturation (Memory Throughput jumped from 45% to 95%) and severe shared memory bank conflicts (2.7-way average, 25% speedup opportunity lost). Although occupancy improved dramatically (12.5% → 75%) and registers/thread dropped (184 → 40), the kernel became 2x slower because the bottleneck shifted from register-limited occupancy to L1 bandwidth and shared memory conflicts, with Warp Cycles Per Issued Instruction increasing from 4.1 to 19.4 cycles due to MIO queue stalls. The total instruction count also ballooned from 1.48B to 3.62B due to the much larger grid (65K → 4.2M blocks), indicating excessive per-block overhead.

> **Applicable when**: Register pressure is the confirmed hard bottleneck (Block Limit Registers = 1, registers/thread > 128, theoretical occupancy < 20%), AND the problem has a large K dimension that benefits from tiling, AND L1/shared memory bandwidth is not already near saturation.

> **Avoid when**: The matrix K dimension is tiny (K <= 32) causing the K-loop to run only 1-2 iterations regardless of BLOCK_K — in this case, smaller tiles create a massive grid explosion (65K → 4.2M blocks) with per-block overhead dominating. Also avoid when L1/TEX throughput is already above 60% or when shared memory access patterns would cause bank conflicts with smaller tiles. Do not apply when the resulting grid size increase causes total instruction count to grow by more than 2x.

---

## Iteration 2

**Best so far**: 2.475× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | skinny_matmul_smem_padding_and_config_pruning_for_bank_conflict_elimination | 1.487× | 1.399× | ✓ |  |
| B | reduce_shared_memory_via_lower_num_stages_to_lift_occupancy | 1.529× | 1.868× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_swizzled_layout | 1.237× | 1.034× | ✓ |  |
| D | large_block_m_skinny_matmul_with_pipeline_stages | 0.808× | 1.067× | ✓ |  |
| E | add_triton_autotune | 1.879× | 2.460× | ✓ |  |


### Metric Comparison

| Metric | Baseline | skinny_matmul_sme... | Δ | reduce_shared_mem... | Δ | eliminate_shared_... | Δ | large_block_m_ski... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3960000 ns | 2830000 ns | -1130000 ns (-28.5%) | 2120000 ns | -1840000 ns (-46.5%) | 3830000 ns | -130000 ns (-3.3%) | 3710000 ns | -250000 ns (-6.3%) | 1610000 ns | -2350000 ns (-59.3%) |
| SM Throughput (%) | 47.34 % | 38.62 % | -8.72 pp | 42.14 % | -5.20 pp | 36.34 % | -11.00 pp | 59.74 % | +12.40 pp | 32.15 % | -15.19 pp |
| Memory Throughput (%) | 45.55 % | 82.25 % | +36.70 pp | 63.67 % | +18.12 pp | 69.08 % | +23.53 pp | 85.85 % | +40.30 pp | 69.18 % | +23.63 pp |
| DRAM Throughput (%) | 22.67 % | 32.39 % | +9.72 pp | 42.31 % | +19.64 pp | 23.42 % | +0.75 pp | 24.19 % | +1.52 pp | 55.64 % | +32.97 pp |
| Achieved Occupancy (%) | 12.41 % | 23.35 % | +10.94 pp | 23.77 % | +11.36 pp | 11.93 % | -0.48 pp | 18.15 % | +5.74 pp | 23.63 % | +11.22 pp |
| Registers/Thread | 184.00 register/thread | 105.00 register/thread | -79 | 128.00 register/thread | -56 | 80.00 register/thread | -104 | 148.00 register/thread | -36 | 125.00 register/thread | -59 |
| Waves/SM | 496.48 | 992.97 | +496.49 | 496.48 | +0.00 | 992.97 | +496.49 | 661.98 | +165.50 | 248.24 | -248.24 |
| No Eligible (%) | 51.48 % | 61.41 % | +9.93 pp | 58.63 % | +7.15 pp | 65.26 % | +13.78 pp | 40.21 % | -11.27 pp | 67.62 % | +16.14 pp |
| IPC Active | 1.93 inst/cycle | 1.55 inst/cycle | -0.38 | 1.66 inst/cycle | -0.27 | 1.38 inst/cycle | -0.55 | 2.39 inst/cycle | +0.46 | 1.29 inst/cycle | -0.64 |


### Key Learnings

#### ✅ Plan D: `large_block_m_skinny_matmul_with_pipeline_stages` (1.067×)

For skinny matmul kernels (large M, small N and K), replacing small-tile autotune configs with large BLOCK_M configs (128–1024) aligned to the exact N and K dimensions reduces warp stalls: Warp Cycles Per Issued Instruction dropped from 11.67 to 4.85 and Issue Slots Busy improved from 32% to 60%, indicating better instruction-level throughput. However, the autotuner selected a config (BLOCK_M=128, block_size=128) that caused the grid to expand from 65K to 262K blocks and registers/thread to increase from 125 to 148, which worsened occupancy (25%→18.75%) and increased L1/Mem Busy (69%→86%), partially offsetting the stall reduction and yielding only a modest 1.07x speedup. The SharedMemoryConflicts penalty actually worsened (10.45%→26.65% speedup_pct flagged), suggesting the tile shape selected still has bank conflict issues.

> **Applicable when**: Skinny matmul (M large, N and K small, e.g. N=K=32) where the prior kernel shows high no-eligible-warp stalls (>60%), low Issue Slots Busy (<35%), and Warp Cycles Per Issued Instruction > 10, with the autotuner selecting configs that don't align BLOCK_N to the actual N dimension.

> **Avoid when**: The autotuner selects a large BLOCK_M config that causes registers/thread to increase significantly (>140), worsening occupancy below 20% while also expanding the grid — in this case the grid explosion and register pressure combine to negate the stall reduction benefit. Do not apply when SharedMemoryConflicts are already the dominant bottleneck (speedup_pct > 20%) as this change does not resolve bank conflicts and may worsen them. Also avoid reverting fp16 tl.dot to fp32 tl.dot simultaneously, as this increases register pressure.

---

## Iteration 3

**Best so far**: 2.475× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_eliminate_bank_conflicts | 1.408× | 1.643× | ✓ |  |
| B | smem_padding_to_eliminate_shared_memory_bank_conflicts | 1.479× | 1.737× | ✓ |  |
| C | smem_bank_conflict_elimination_via_b_tile_transpose_and_k_loop_restructure | 0.867× | 1.067× | ✓ |  |
| D | single_iteration_k_loop_num_stages_one_with_wider_block_m_and_warp_sweep | 1.329× | 1.744× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | smem_padding_to_e... | Δ | smem_bank_conflic... | Δ | single_iteration_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3960000 ns | 2410000 ns | -1550000 ns (-39.1%) | 2280000 ns | -1680000 ns (-42.4%) | 3710000 ns | -250000 ns (-6.3%) | 2270000 ns | -1690000 ns (-42.7%) |
| SM Throughput (%) | 47.34 % | 66.16 % | +18.82 pp | 46.85 % | -0.49 pp | 59.74 % | +12.40 pp | 27.43 % | -19.91 pp |
| Memory Throughput (%) | 45.55 % | 89.69 % | +44.14 pp | 57.53 % | +11.98 pp | 85.86 % | +40.31 pp | 61.55 % | +16.00 pp |
| DRAM Throughput (%) | 22.67 % | 37.94 % | +15.27 pp | 39.36 % | +16.69 pp | 24.19 % | +1.52 pp | 39.57 % | +16.90 pp |
| Achieved Occupancy (%) | 12.41 % | 41.83 % | +29.42 pp | 23.86 % | +11.45 pp | 18.15 % | +5.74 pp | 11.90 % | -0.51 pp |
| Registers/Thread | 184.00 register/thread | 71.00 register/thread | -113 | 101.00 register/thread | -83 | 148.00 register/thread | -36 | 174.00 register/thread | -10 |
| Waves/SM | 496.48 | 567.41 | +70.93 | 496.48 | +0.00 | 661.98 | +165.50 | 496.48 | +0.00 |
| No Eligible (%) | 51.48 % | 33.76 % | -17.72 pp | 52.84 % | +1.36 pp | 40.21 % | -11.27 pp | 76.27 % | +24.79 pp |
| IPC Active | 1.93 inst/cycle | 2.65 inst/cycle | +0.72 | 1.88 inst/cycle | -0.05 | 2.39 inst/cycle | +0.46 | 0.94 inst/cycle | -0.99 |

---

## Cumulative Progress

```
Baseline:  3960000 ns  (1.000×)
Iter 1     1600000 ns  (2.475×)  — fast_accumulation_fp16_dot_with_reduced_tile_sizes

Final best: 1600000 ns  (2.4750×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -13,6 +14,19 @@

         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
+        # Smaller tile configs to reduce register pressure and raise occupancy
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
+        # New configs tuned for N=32, K=32: BLOCK_N=32 matches actual N, num_stages=1 since K-loop runs once
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -61,7 +75,8 @@

         b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
         b = tl.load(b_ptrs, mask=b_mask, other=0.0)
 
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
+        # Cast inputs to FP16 to activate Tensor Cores; accumulator stays FP32
+        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc=acc, out_dtype=tl.float32)
 
         a_ptrs += BLOCK_K * stride_ak
         b_ptrs += BLOCK_K * stride_bk
@@ -72,7 +87,6 @@

     c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
     tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)
-
 
 def kernel_function(a, b):
     # a: (M, K), b: (K, N)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
