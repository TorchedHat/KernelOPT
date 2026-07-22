# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_native_layer_norm_t_view_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 100800 ns — **Best latency**: 100900 ns (**0.9990×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_native_layer_norm_t_view_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_15_triton_tem_fused_mm_native_lay/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (246,1,1), Waves/SM: 0.93
- **Registers/thread**: 255
- **Duration**: 100800 ns
- **SM Throughput**: 35.21 % | **Memory**: 79.84 % | **DRAM**: 2.52 %
- **Achieved Occupancy**: 11.82 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 62.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.003× | 0.998× | ✓ | **✓ best** |
| C | add_eviction_policies_for_streaming_loads | 1.001× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 100800 ns | 100960 ns | +160 ns (+0.2%) | 100900 ns | +100 ns (+0.1%) |
| SM Throughput (%) | 35.21 % | 35.08 % | -0.13 pp | 35.00 % | -0.21 pp |
| Memory Throughput (%) | 79.84 % | 79.53 % | -0.31 pp | 79.35 % | -0.49 pp |
| DRAM Throughput (%) | 2.52 % | 2.51 % | -0.01 pp | 2.51 % | -0.01 pp |
| Achieved Occupancy (%) | 11.82 % | 11.82 % | +0.00 pp | 11.82 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 62.45 % | 62.44 % | -0.01 pp | 62.27 % | -0.18 pp |
| IPC Active | 1.50 inst/cycle | 1.49 inst/cycle | -0.01 | 1.51 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  100800 ns  (1.000×)

Final best: 100900 ns  (0.9990×)
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
 def triton_tem_fused_mm_native_layer_norm_t_view_15(arg_A, arg_B, out_ptr0):
@@ -65,12 +66,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 384*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +86,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 192*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
