# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_gelu_t_view_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 260030 ns — **Best latency**: 260030 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_gelu_t_view_37`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_37_triton_tem_fused_addmm_gelu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (93,1,1), Waves/SM: 0.23
- **Registers/thread**: 168
- **Duration**: 260030 ns
- **SM Throughput**: 19.93 % | **Memory**: 44.89 % | **DRAM**: 1.13 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.22 inst/cycle | **No Eligible**: 69.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_tile_and_warp_sizing | 0.879× | 0.870× | ✓ |  |
| C | autotune_tile_sizes_with_eviction_policies | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 260030 ns | 299040 ns | +39010 ns (+15.0%) | 260030 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 19.93 % | 17.36 % | -2.57 pp | 19.92 % | -0.01 pp |
| Memory Throughput (%) | 44.89 % | 39.15 % | -5.74 pp | 44.86 % | -0.03 pp |
| DRAM Throughput (%) | 1.13 % | 0.99 % | -0.14 pp | 1.13 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.35 | +0.12 | 0.23 | +0.00 |
| No Eligible (%) | 69.58 % | 72.99 % | +3.41 pp | 69.58 % | +0.00 pp |
| IPC Active | 1.22 inst/cycle | 1.08 inst/cycle | -0.14 | 1.22 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 0.992× | 0.990× | ✓ |  |
| B | fix_b_matrix_noncoalesced_broadcast_load | 0.994× | 0.994× | ✓ |  |
| C | eliminate_tl_trans_bank_conflicts_via_b_layout_rewrite | 0.899× | 0.856× | ✓ |  |
| D | reduce_register_pressure_via_smaller_block_k_and_maxnreg_cap | 0.823× | 0.821× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | fix_b_matrix_nonc... | Δ | eliminate_tl_tran... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 260030 ns | 262750 ns | +2720 ns (+1.0%) | 261630 ns | +1600 ns (+0.6%) | 303780 ns | +43750 ns (+16.8%) | 316830 ns | +56800 ns (+21.8%) |
| SM Throughput (%) | 19.93 % | 19.72 % | -0.21 pp | 19.77 % | -0.16 pp | 17.11 % | -2.82 pp | 18.31 % | -1.62 pp |
| Memory Throughput (%) | 44.89 % | 44.40 % | -0.49 pp | 44.54 % | -0.35 pp | 38.86 % | -6.03 pp | 39.69 % | -5.20 pp |
| DRAM Throughput (%) | 1.13 % | 1.12 % | -0.01 pp | 1.12 % | -0.01 pp | 0.97 % | -0.16 pp | 0.93 % | -0.20 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 | 128.00 register/thread | -40 |
| Waves/SM | 0.23 | 0.23 | +0.00 | 0.23 | +0.00 | 0.35 | +0.12 | 0.18 | -0.05 |
| No Eligible (%) | 69.58 % | 69.75 % | +0.17 pp | 69.72 % | +0.14 pp | 73.65 % | +4.07 pp | 71.61 % | +2.03 pp |
| IPC Active | 1.22 inst/cycle | 1.21 inst/cycle | -0.01 | 1.21 inst/cycle | -0.01 | 1.06 inst/cycle | -0.16 | 1.13 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  260030 ns  (1.000×)

Final best: 260030 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused_addmm_gelu_t_view_37(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 1536*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 384*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +87,6 @@

     # inductor generates a suffix
     xindex = idx_n + 384*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
