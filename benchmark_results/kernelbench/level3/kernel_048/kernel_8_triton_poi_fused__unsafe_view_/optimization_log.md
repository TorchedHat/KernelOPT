# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 646050 ns — **Best latency**: 646050 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_8_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 16
- **Duration**: 646050 ns
- **SM Throughput**: 34.55 % | **Memory**: 33.23 % | **DRAM**: 33.23 %
- **Achieved Occupancy**: 42.59 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_grid_and_hide_memory_latency | 0.878× | 0.887× | ✓ |  |
| B | increase_xblock_tile_size_to_reduce_latency_and_improve_parallelism | 0.996× | 0.509× | ✓ |  |
| D | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.878× | 0.889× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 646050 ns | 728350 ns | +82300 ns (+12.7%) | 1270000 ns | +623950 ns (+96.6%) | 726880 ns | +80830 ns (+12.5%) |
| SM Throughput (%) | 34.55 % | 35.26 % | +0.71 pp | 80.69 % | +46.14 pp | 35.38 % | +0.83 pp |
| Memory Throughput (%) | 33.23 % | 29.36 % | -3.87 pp | 17.08 % | -16.15 pp | 29.43 % | -3.80 pp |
| DRAM Throughput (%) | 33.23 % | 29.36 % | -3.87 pp | 16.83 % | -16.40 pp | 29.43 % | -3.80 pp |
| Achieved Occupancy (%) | 42.59 % | 34.16 % | -8.43 pp | 66.55 % | +23.96 pp | 34.27 % | -8.32 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 1985.94 | +1489.46 | 496.48 | +0.00 |
| No Eligible (%) | 71.20 % | 68.63 % | -2.57 pp | 35.59 % | -35.61 pp | 69.06 % | -2.14 pp |
| IPC Active | 1.15 inst/cycle | 1.21 inst/cycle | +0.06 | 2.57 inst/cycle | +1.42 | 1.21 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.999× | 0.999× | ✓ |  |
| B | persistent_tile_scheduling | 1.000× | 0.999× | ✓ |  |
| C | fix_autotune_grid_and_reduce_xblock_to_eliminate_alu_index_arithmetic_blowup | 1.000× | 1.000× | ✓ |  |
| D | add_triton_autotune | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 646050 ns | 646430 ns | +380 ns (+0.1%) | 646430 ns | +380 ns (+0.1%) | 646110 ns | +60 ns (+0.0%) | 646180 ns | +130 ns (+0.0%) |
| SM Throughput (%) | 34.55 % | 40.97 % | +6.42 pp | 36.04 % | +1.49 pp | 34.44 % | -0.11 pp | 36.19 % | +1.64 pp |
| Memory Throughput (%) | 33.23 % | 33.21 % | -0.02 pp | 33.21 % | -0.02 pp | 33.23 % | +0.00 pp | 33.22 % | -0.01 pp |
| DRAM Throughput (%) | 33.23 % | 33.21 % | -0.02 pp | 33.21 % | -0.02 pp | 33.23 % | +0.00 pp | 33.22 % | -0.01 pp |
| Achieved Occupancy (%) | 42.59 % | 46.93 % | +4.34 pp | 45.45 % | +2.86 pp | 42.62 % | +0.03 pp | 43.08 % | +0.49 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 71.20 % | 61.33 % | -9.87 pp | 66.26 % | -4.94 pp | 71.19 % | -0.01 pp | 66.27 % | -4.93 pp |
| IPC Active | 1.15 inst/cycle | 1.55 inst/cycle | +0.40 | 1.35 inst/cycle | +0.20 | 1.15 inst/cycle | +0.00 | 1.35 inst/cycle | +0.20 |

---

## Cumulative Progress

```
Baseline:  646050 ns  (1.000×)

Final best: 646050 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,27 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 134217728
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 64)
     x1 = ((xindex // 64) % 64)
     x2 = xindex // 4096
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 64*((x2 % 8)) + 512*x1 + 32768*(x2 // 8)), None)
-    tl.store(out_ptr0 + (x3), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 64*((x2 % 8)) + 512*x1 + 32768*(x2 // 8)), xmask)
+    tl.store(out_ptr0 + (x3), tmp0, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 134217728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8[grid](arg_0, arg_1, 134217728, XBLOCK=128)
+    triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8[grid](arg_0, arg_1, 134217728)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
