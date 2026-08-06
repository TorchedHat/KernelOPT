# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17860 ns — **Best latency**: 17860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_004/kernel_6_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 17860 ns
- **SM Throughput**: 18.87 % | **Memory**: 70.30 % | **DRAM**: 2.32 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.89 inst/cycle | **No Eligible**: 77.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.990× | 0.998× | ✓ |  |
| C | eviction_policies_streaming_loads | 0.990× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 17860 ns | 17890 ns | +30 ns (+0.2%) | 17660 ns | -200 ns (-1.1%) |
| SM Throughput (%) | 18.87 % | 18.90 % | +0.03 pp | 18.46 % | -0.41 pp |
| Memory Throughput (%) | 70.30 % | 70.42 % | +0.12 pp | 68.69 % | -1.61 pp |
| DRAM Throughput (%) | 2.32 % | 2.32 % | +0.00 pp | 2.35 % | +0.03 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.70 % | 77.67 % | -0.03 pp | 77.71 % | +0.01 pp |
| IPC Active | 0.89 inst/cycle | 0.87 inst/cycle | -0.02 | 0.89 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_num_warps_and_num_stages | 0.991× | 0.787× | ✓ |  |
| C | reduce_register_pressure_via_num_warps_and_pipeline_stages | 0.990× | 1.006× | ✓ |  |
| D | fix_index_arithmetic_and_autotune_tile_sizes | 0.989× | 0.814× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | fix_index_arithme... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17860 ns | 22690 ns | +4830 ns (+27.0%) | 17760 ns | -100 ns (-0.6%) | 21950 ns | +4090 ns (+22.9%) |
| SM Throughput (%) | 18.87 % | 14.08 % | -4.79 pp | 18.66 % | -0.21 pp | 14.63 % | -4.24 pp |
| Memory Throughput (%) | 70.30 % | 33.68 % | -36.62 pp | 69.52 % | -0.78 pp | 32.62 % | -37.68 pp |
| DRAM Throughput (%) | 2.32 % | 1.85 % | -0.47 pp | 2.34 % | +0.02 pp | 1.91 % | -0.41 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 128.00 register/thread | 254.00 register/thread | +126 | 128.00 register/thread | +0 | 218.00 register/thread | +90 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 77.70 % | 66.10 % | -11.60 pp | 77.71 % | +0.01 pp | 63.54 % | -14.16 pp |
| IPC Active | 0.89 inst/cycle | 0.68 inst/cycle | -0.21 | 0.89 inst/cycle | +0.00 | 0.73 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  17860 ns  (1.000×)

Final best: 17860 ns  (1.0000×)
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
 def triton_tem_fused_addmm_relu_t_6(arg_A, arg_B, out_ptr0):
@@ -68,12 +69,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 120*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        a = tl.load(A + (xindex), mask=a_mask, other=0.0, eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 84*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 120*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 120*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0, eviction_policy='evict_first')
 
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -88,9 +89,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 84*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
