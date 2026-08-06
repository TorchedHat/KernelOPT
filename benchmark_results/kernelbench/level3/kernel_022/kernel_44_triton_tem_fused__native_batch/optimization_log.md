# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 58020 ns — **Best latency**: 58080 ns (**0.9990×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_44_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 168
- **Duration**: 58020 ns
- **SM Throughput**: 16.32 % | **Memory**: 36.60 % | **DRAM**: 0.81 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 70.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_make_block_ptr | 0.992× | 0.985× | ✓ |  |
| C | add_eviction_policies_streaming_loads | 1.004× | 0.999× | ✓ | **✓ best** |
| D | fix_b_matrix_load_indexing_and_increase_tile_sizes | 0.931× | 0.866× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_mak... | Δ | add_eviction_poli... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 58020 ns | 58910 ns | +890 ns (+1.5%) | 58080 ns | +60 ns (+0.1%) | 67010 ns | +8990 ns (+15.5%) |
| SM Throughput (%) | 16.32 % | 16.12 % | -0.20 pp | 16.27 % | -0.05 pp | 14.35 % | -1.97 pp |
| Memory Throughput (%) | 36.60 % | 36.15 % | -0.45 pp | 36.49 % | -0.11 pp | 32.37 % | -4.23 pp |
| DRAM Throughput (%) | 0.81 % | 0.80 % | -0.01 pp | 0.81 % | +0.00 pp | 0.72 % | -0.09 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.20 | 0.20 | +0.00 | 0.20 | +0.00 | 0.30 | +0.10 |
| No Eligible (%) | 70.06 % | 70.20 % | +0.14 pp | 70.03 % | -0.03 pp | 73.84 % | +3.78 pp |
| IPC Active | 1.20 inst/cycle | 1.19 inst/cycle | -0.01 | 1.19 inst/cycle | -0.01 | 1.04 inst/cycle | -0.16 |


### Key Learnings

#### ❌ Plan D: `fix_b_matrix_load_indexing_and_increase_tile_sizes` (0.866×)

Increasing BLOCK_K from 32 to 64 in a Triton GEMM kernel caused registers per thread to jump from 168 to 255, which reduced Block Limit Registers from 3 to 2 and theoretical occupancy from 18.75% to 12.5%. Despite halving k-loop iterations, the increased register pressure worsened warp cycles per issued instruction (3.34 → 3.84) and raised elapsed cycles by ~15%, resulting in a 13% slowdown. When a kernel is already register-limited (Block Limit Registers ≤ 3), increasing tile sizes along the K dimension further inflates register usage and compounds the occupancy bottleneck rather than improving compute intensity.

> **Applicable when**: Never: increasing BLOCK_K when already register-limited causes regression. This is a negative example.

> **Avoid when**: When Block Limit Registers ≤ 3 and registers_per_thread ≥ 128, increasing BLOCK_K will further inflate register usage, reduce theoretical occupancy, and increase warp stall cycles. Do not increase any tile dimension when the kernel is already register-limited.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_b_matrix_load_indexing_eliminate_bank_conflicts | — | fail | ✓ |  |
| D | fix_b_matrix_broadcast_indexing_bug_to_eliminate_bank_conflicts | — | fail | ✓ |  |

---

## Cumulative Progress

```
Baseline:  58020 ns  (1.000×)

Final best: 58080 ns  (0.9990×)
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
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 320*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 1280*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 320*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 320*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
 
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 1280*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, cache_modifier='.wb')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
