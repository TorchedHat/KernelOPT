# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_native_layer_norm_t_view_50` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 167780 ns — **Best latency**: 176540 ns (**0.9504×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_native_layer_norm_t_view_50`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_50_triton_tem_fused_mm_native_lay/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.18
- **Registers/thread**: 128
- **Duration**: 167780 ns
- **SM Throughput**: 16.37 % | **Memory**: 64.30 % | **DRAM**: 0.94 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_to_fix_bank_conflicts_and_occupancy | 0.091× | 0.085× | ✓ |  |
| C | add_autotune_reduce_register_pressure_fix_bank_conflicts | 1.042× | 0.950× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 167780 ns | 1980000 ns | +1812220 ns (+1080.1%) | 176540 ns | +8760 ns (+5.2%) |
| SM Throughput (%) | 16.37 % | 6.58 % | -9.79 pp | 15.86 % | -0.51 pp |
| Memory Throughput (%) | 64.30 % | 13.49 % | -50.81 pp | 61.49 % | -2.81 pp |
| DRAM Throughput (%) | 0.94 % | 5.85 % | +4.91 pp | 0.89 % | -0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 255.00 register/thread | +127 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.73 | +0.55 | 0.18 | +0.00 |
| No Eligible (%) | 76.88 % | 89.60 % | +12.72 pp | 77.87 % | +0.99 pp |
| IPC Active | 0.92 inst/cycle | 0.42 inst/cycle | -0.50 | 0.88 inst/cycle | -0.04 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_to_reduce_shared_memory_bank_conflicts | 0.996× | 0.999× | ✓ |  |
| B | increase_block_k_to_reduce_k_loop_overhead_and_improve_dot_efficiency | 0.999× | 1.000× | ✓ |  |
| C | revert_to_scalar_pointer_arithmetic_with_small_tiles_to_fix_register_smem_explosion | 0.962× | 0.964× | ✓ |  |
| D | reduce_tile_sizes_to_fix_register_spill_and_restore_occupancy | 0.936× | 0.935× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | increase_block_k_... | Δ | revert_to_scalar_... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 167780 ns | 167900 ns | +120 ns (+0.1%) | 167810 ns | +30 ns (+0.0%) | 174080 ns | +6300 ns (+3.8%) | 179390 ns | +11610 ns (+6.9%) |
| SM Throughput (%) | 16.37 % | 16.46 % | +0.09 pp | 16.38 % | +0.01 pp | 16.06 % | -0.31 pp | 15.63 % | -0.74 pp |
| Memory Throughput (%) | 64.30 % | 64.68 % | +0.38 pp | 64.36 % | +0.06 pp | 62.49 % | -1.81 pp | 60.38 % | -3.92 pp |
| DRAM Throughput (%) | 0.94 % | 0.94 % | +0.00 pp | 0.94 % | +0.00 pp | 0.91 % | -0.03 pp | 0.88 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 76.88 % | 76.91 % | +0.03 pp | 76.90 % | +0.02 pp | 77.58 % | +0.70 pp | 78.20 % | +1.32 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.92 inst/cycle | +0.00 | 0.90 inst/cycle | -0.02 | 0.87 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_index_to_eliminate_shared_memory_bank_conflicts | 0.998× | 0.998× | ✓ |  |
| B | fix_b_matrix_access_pattern_via_transposed_load | 0.924× | 0.925× | ✓ |  |
| C | fix_b_matrix_load_coalescing_to_eliminate_shared_memory_bank_conflicts | 1.000× | 0.998× | ✓ |  |
| D | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 0.935× | 0.935× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_acce... | Δ | fix_b_matrix_load... | Δ | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 167780 ns | 168160 ns | +380 ns (+0.2%) | 181470 ns | +13690 ns (+8.2%) | 168160 ns | +380 ns (+0.2%) | 179460 ns | +11680 ns (+7.0%) |
| SM Throughput (%) | 16.37 % | 16.53 % | +0.16 pp | 15.59 % | -0.78 pp | 16.38 % | +0.01 pp | 15.63 % | -0.74 pp |
| Memory Throughput (%) | 64.30 % | 64.95 % | +0.65 pp | 59.96 % | -4.34 pp | 64.36 % | +0.06 pp | 60.37 % | -3.93 pp |
| DRAM Throughput (%) | 0.94 % | 0.96 % | +0.02 pp | 0.87 % | -0.07 pp | 0.94 % | +0.00 pp | 0.88 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 76.88 % | 76.91 % | +0.03 pp | 78.09 % | +1.21 pp | 76.87 % | -0.01 pp | 78.21 % | +1.33 pp |
| IPC Active | 0.92 inst/cycle | 0.92 inst/cycle | +0.00 | 0.87 inst/cycle | -0.05 | 0.92 inst/cycle | +0.00 | 0.87 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  167780 ns  (1.000×)

Final best: 176540 ns  (0.9504×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_mm_native_layer_norm_t_view_50(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,16 +79,14 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 1536*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 768*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +98,6 @@

     # inductor generates a suffix
     xindex = idx_n + 768*idx_m
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
