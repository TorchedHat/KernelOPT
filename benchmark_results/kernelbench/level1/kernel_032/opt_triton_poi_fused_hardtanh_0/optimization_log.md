# Kernel Optimization Log

## Summary

- **Kernel**: `hardtanh_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 2990000 ns (**1.0100×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_pressure_via_smaller_block_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `hardtanh_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_032/synthesis/kernel_0_triton_poi_fused_hardtanh_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 18
- **Duration**: 3020000 ns
- **SM Throughput**: 12.18 % | **Memory**: 86.56 % | **DRAM**: 86.56 %
- **Achieved Occupancy**: 85.22 %
- **IPC Active**: 0.43 inst/cycle | **No Eligible**: 89.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_l2 | 0.978× | 0.993× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 1.001× | 0.981× | ✓ |  |
| C | increase_block_size_and_pipeline_stages_for_memory_bound_pointwise | 1.001× | 0.974× | ✓ |  |
| D | increase_block_size_and_add_software_pipelining | 1.011× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_block_si... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3080000 ns | +60000 ns (+2.0%) | 3100000 ns | +80000 ns (+2.6%) | 2990000 ns | -30000 ns (-1.0%) |
| SM Throughput (%) | 12.18 % | 13.45 % | +1.27 pp | 12.37 % | +0.19 pp | 27.99 % | +15.81 pp | 28.82 % | +16.64 pp |
| Memory Throughput (%) | 86.56 % | 85.96 % | -0.60 pp | 84.87 % | -1.69 pp | 84.36 % | -2.20 pp | 87.50 % | +0.94 pp |
| DRAM Throughput (%) | 86.56 % | 85.96 % | -0.60 pp | 84.87 % | -1.69 pp | 84.36 % | -2.20 pp | 87.50 % | +0.94 pp |
| Achieved Occupancy (%) | 85.22 % | 85.16 % | -0.06 pp | 72.23 % | -12.99 pp | 20.53 % | -64.69 pp | 20.65 % | -64.57 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 90.00 register/thread | +72 | 90.00 register/thread | +72 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 372.36 | -372.37 | 372.36 | -372.37 |
| No Eligible (%) | 89.20 % | 86.49 % | -2.71 pp | 88.88 % | -0.32 pp | 83.06 % | -6.14 pp | 80.90 % | -8.30 pp |
| IPC Active | 0.43 inst/cycle | 0.54 inst/cycle | +0.11 | 0.45 inst/cycle | +0.02 | 0.66 inst/cycle | +0.23 | 0.75 inst/cycle | +0.32 |

---

## Iteration 2

**Best so far**: 1.01× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_smaller_block_sizes | 1.016× | 1.010× | ✓ | **✓ best** |
| B | l2_compression_via_bf16_output_and_larger_autotune_configs | 0.964× | 0.968× | ✓ |  |
| C | increase_num_stages_for_latency_hiding | 1.001× | 0.981× | ✓ |  |
| D | reduce_register_pressure_via_fewer_warps_per_block | 1.001× | 0.971× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | l2_compression_vi... | Δ | increase_num_stag... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 2990000 ns | -30000 ns (-1.0%) | 3120000 ns | +100000 ns (+3.3%) | 3080000 ns | +60000 ns (+2.0%) | 3110000 ns | +90000 ns (+3.0%) |
| SM Throughput (%) | 12.18 % | 28.80 % | +16.62 pp | 21.99 % | +9.81 pp | 12.45 % | +0.27 pp | 28.22 % | +16.04 pp |
| Memory Throughput (%) | 86.56 % | 87.53 % | +0.97 pp | 83.93 % | -2.63 pp | 84.88 % | -1.68 pp | 84.23 % | -2.33 pp |
| DRAM Throughput (%) | 86.56 % | 87.53 % | +0.97 pp | 83.93 % | -2.63 pp | 84.88 % | -1.68 pp | 84.23 % | -2.33 pp |
| Achieved Occupancy (%) | 85.22 % | 20.60 % | -64.62 pp | 38.79 % | -46.43 pp | 72.28 % | -12.94 pp | 20.50 % | -64.72 pp |
| Registers/Thread | 18.00 register/thread | 90.00 register/thread | +72 | 52.00 register/thread | +34 | 22.00 register/thread | +4 | 90.00 register/thread | +72 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 372.36 | -372.37 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 89.20 % | 80.75 % | -8.45 pp | 83.77 % | -5.43 pp | 88.84 % | -0.36 pp | 82.92 % | -6.28 pp |
| IPC Active | 0.43 inst/cycle | 0.76 inst/cycle | +0.33 | 0.64 inst/cycle | +0.21 | 0.44 inst/cycle | +0.01 | 0.66 inst/cycle | +0.23 |

---

## Iteration 3

**Best so far**: 1.01× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_load_eviction_policy_and_add_cta_cluster_configs | 0.978× | 0.962× | ✓ |  |
| C | streaming_cache_eviction_policy_for_single_use_data | 1.000× | 0.962× | ✓ |  |
| D | reduce_register_pressure_via_fewer_warps_larger_block_configs | 0.965× | 0.965× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_load_eviction... | Δ | streaming_cache_e... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3140000 ns | +120000 ns (+4.0%) | 3140000 ns | +120000 ns (+4.0%) | 3130000 ns | +110000 ns (+3.6%) |
| SM Throughput (%) | 12.18 % | 27.88 % | +15.70 pp | 27.58 % | +15.40 pp | 28.36 % | +16.18 pp |
| Memory Throughput (%) | 86.56 % | 83.33 % | -3.23 pp | 83.41 % | -3.15 pp | 83.53 % | -3.03 pp |
| DRAM Throughput (%) | 86.56 % | 83.33 % | -3.23 pp | 83.41 % | -3.15 pp | 83.53 % | -3.03 pp |
| Achieved Occupancy (%) | 85.22 % | 20.78 % | -64.44 pp | 20.52 % | -64.70 pp | 28.81 % | -56.41 pp |
| Registers/Thread | 18.00 register/thread | 90.00 register/thread | +72 | 90.00 register/thread | +72 | 87.00 register/thread | +69 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 372.36 | -372.37 | 297.89 | -446.84 |
| No Eligible (%) | 89.20 % | 81.76 % | -7.44 pp | 83.26 % | -5.94 pp | 81.98 % | -7.22 pp |
| IPC Active | 0.43 inst/cycle | 0.74 inst/cycle | +0.31 | 0.66 inst/cycle | +0.23 | 0.72 inst/cycle | +0.29 |

---

## Iteration 4

**Best so far**: 1.01× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_and_expand_warp_configs | 0.964× | 0.508× | ✓ |  |
| B | fp16_intermediate_clamp_with_low_warp_configs_for_occupancy | 0.978× | 0.944× | ✓ |  |
| C | add_num_warps_1_configs_to_reduce_register_pressure_and_increase_warp_parallelism | 1.000× | 0.962× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | fp16_intermediate... | Δ | add_num_warps_1_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 5940000 ns | +2920000 ns (+96.7%) | 3200000 ns | +180000 ns (+6.0%) | 3140000 ns | +120000 ns (+4.0%) |
| SM Throughput (%) | 12.18 % | 15.07 % | +2.89 pp | 11.48 % | -0.70 pp | 13.53 % | +1.35 pp |
| Memory Throughput (%) | 86.56 % | 72.73 % | -13.83 pp | 81.72 % | -4.84 pp | 83.21 % | -3.35 pp |
| DRAM Throughput (%) | 86.56 % | 72.73 % | -13.83 pp | 81.72 % | -4.84 pp | 83.21 % | -3.35 pp |
| Achieved Occupancy (%) | 85.22 % | 70.75 % | -14.47 pp | 32.43 % | -52.79 pp | 19.88 % | -65.34 pp |
| Registers/Thread | 18.00 register/thread | 40.00 register/thread | +22 | 76.00 register/thread | +58 | 91.00 register/thread | +73 |
| Waves/SM | 744.73 | 124.12 | -620.61 | 248.24 | -496.49 | 372.36 | -372.37 |
| No Eligible (%) | 89.20 % | 87.53 % | -1.67 pp | 88.90 % | -0.30 pp | 91.19 % | +1.99 pp |
| IPC Active | 0.43 inst/cycle | 0.50 inst/cycle | +0.07 | 0.44 inst/cycle | +0.01 | 0.35 inst/cycle | -0.08 |


### Key Learnings

#### ❌ Plan A: `cap_registers_with_maxnreg_and_expand_warp_configs` (0.508×)

Capping registers via maxnreg=40 successfully raised theoretical occupancy from 31.25% to 75% and achieved occupancy from 28.81% to 70.75%, but caused a 2x slowdown (3.13ms → 5.94ms). The register cap forced register spills to local memory, as evidenced by new MemoryCacheAccessPattern warnings (only 1 of 32 bytes per sector utilized for local loads/stores), a 3.5x increase in Warp Cycles Per Issued Instruction (25.66 → 90.74), a 2.4x increase in L2 Compression Input Sectors (201M → 487M), and a drop in DRAM throughput from 83.53% to 72.73%. More active warps (higher occupancy) amplified the spill traffic, overwhelming the memory subsystem. Do not apply maxnreg register capping when the kernel's actual register demand is high relative to the cap value — forced spills to local memory create strided, uncoalesced access patterns that are far more costly than the latency-hiding benefit of higher occupancy.

> **Applicable when**: Never applicable when the kernel's natural register count significantly exceeds the cap target — spill overhead will dominate. Only consider maxnreg when the compiler's actual register usage is already near the cap (within ~10-15 registers), so spilling is minimal.

> **Avoid when**: Kernel has high natural register usage (e.g., 87 regs/thread) and the cap target is aggressively low (e.g., 40 regs/thread). The resulting local memory spills create uncoalesced strided access patterns (1 of 32 bytes per sector utilized), causing Warp Cycles Per Issued Instruction to balloon (25→90 cycles) and net throughput to drop despite higher occupancy. Also avoid when the kernel is already near DRAM saturation (>80% DRAM throughput) — adding more in-flight warps via occupancy increase will not improve throughput if HBM is already the bottleneck.

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)
Iter 1     2990000 ns  (1.010×)  — increase_block_size_and_add_software_pipelining

Final best: 2990000 ns  (1.0100×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,23 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
     ],
     key=['n_elements'],
 )
@@ -27,12 +36,11 @@

     block_start = pid * BLOCK_SIZE
     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
-    x = tl.load(input_ptr + offsets, mask=mask)
+    x = tl.load(input_ptr + offsets, mask=mask, eviction_policy='evict_first')
     # HardTanh: clamp to [min_val, max_val]
     x = tl.where(x < min_val, min_val, x)
     x = tl.where(x > max_val, max_val, x)
-    tl.store(output_ptr + offsets, x, mask=mask)
-
+    tl.store(output_ptr + offsets, x, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
