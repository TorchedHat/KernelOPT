# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 647620 ns — **Best latency**: 648030 ns (**0.9994×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_048/kernel_16_triton_poi_fused_add_clone_exp/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 16
- **Duration**: 647620 ns
- **SM Throughput**: 43.60 % | **Memory**: 50.54 % | **DRAM**: 50.54 %
- **Achieved Occupancy**: 51.56 %
- **IPC Active**: 1.75 inst/cycle | **No Eligible**: 56.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_larger_xblock | — | 0.999× | ✓ | **✓ best** |
| B | autotune_xblock_for_latency_hiding_and_cache_reuse | — | 0.765× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 647620 ns | 648030 ns | +410 ns (+0.1%) | 845980 ns | +198360 ns (+30.6%) |
| SM Throughput (%) | 43.60 % | 43.74 % | +0.14 pp | 55.90 % | +12.30 pp |
| Memory Throughput (%) | 50.54 % | 50.51 % | -0.03 pp | 38.58 % | -11.96 pp |
| DRAM Throughput (%) | 50.54 % | 50.51 % | -0.03 pp | 38.58 % | -11.96 pp |
| Achieved Occupancy (%) | 51.56 % | 51.50 % | -0.06 pp | 46.60 % | -4.96 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 25.00 register/thread | +9 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 56.29 % | 56.06 % | -0.23 pp | 44.31 % | -11.98 pp |
| IPC Active | 1.75 inst/cycle | 1.74 inst/cycle | -0.01 | 2.26 inst/cycle | +0.51 |

---

## Cumulative Progress

```
Baseline:  647620 ns  (1.000×)

Final best: 648030 ns  (0.9994×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 134217728
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x4 = xindex
+    x4 = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
     x5 = xindex // 64
     x0 = (xindex % 64)
     x1 = ((xindex // 64) % 64)
@@ -24,8 +25,6 @@

     tmp4 = tmp1 * tmp3
     tmp5 = tmp0 + tmp4
     tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
