# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11970 ns — **Best latency**: 11940 ns (**1.0025×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_increase_tile_size`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14700,1,1), Waves/SM: 6.96
- **Registers/thread**: 18
- **Duration**: 11970 ns
- **SM Throughput**: 55.41 % | **Memory**: 25.59 % | **DRAM**: 12.83 %
- **Achieved Occupancy**: 60.79 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 30.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.014× | 1.002× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.778× | 0.724× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11970 ns | 11940 ns | -30 ns (-0.3%) | 16540 ns | +4570 ns (+38.2%) |
| SM Throughput (%) | 55.41 % | 55.36 % | -0.05 pp | 73.49 % | +18.08 pp |
| Memory Throughput (%) | 25.59 % | 25.56 % | -0.03 pp | 24.45 % | -1.14 pp |
| DRAM Throughput (%) | 12.83 % | 12.88 % | +0.05 pp | 9.28 % | -3.55 pp |
| Achieved Occupancy (%) | 60.79 % | 61.21 % | +0.42 pp | 84.14 % | +23.35 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 6.96 | 6.96 | +0.00 | 6.96 | +0.00 |
| No Eligible (%) | 30.33 % | 30.73 % | +0.40 pp | 17.03 % | -13.30 pp |
| IPC Active | 2.76 inst/cycle | 2.78 inst/cycle | +0.02 | 3.33 inst/cycle | +0.57 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.724×)

Adding @triton.autotune with larger XBLOCK (512–2048) and higher num_warps (4–16) increased registers per thread from 18 to 32, which reduced L1 cache hit rate from 84.9% to 74.5% and nearly doubled executed instructions (5.1M → 9.4M). Although achieved occupancy improved (60.8% → 84.1%) and eligible warps per scheduler rose (1.95 → 6.64), the increased instruction count and register spill pressure caused warp cycles per issued instruction to worsen (13.87 → 16.08 cycles), resulting in a 38% longer kernel duration. When a kernel is already register-efficient (≤18 regs/thread) and has high L1 hit rates for broadcast/parameter loads, enlarging XBLOCK forces the compiler to allocate more registers, degrading cache utilization and negating the latency-hiding benefit of more warps.

> **Applicable when**: Never apply this specific strategy when the baseline kernel already uses a small XBLOCK (≤128) with low register count (≤18 regs/thread) and high L1 hit rate (>80%) for broadcast parameter loads. The CPIStall signal alone is insufficient justification if eligible warps are already >1.5 and L1 hit rate is high.

> **Avoid when**: Kernel has ≤18 registers/thread and L1/TEX hit rate >80%; increasing XBLOCK will cause the compiler to allocate more registers (observed jump to 32), reducing L1 residency for cached broadcast tensors and increasing total instruction count, causing net regression despite improved occupancy.

---

## Cumulative Progress

```
Baseline:  11970 ns  (1.000×)
Iter 1     11940 ns  (1.002×)  — autotune_xblock_increase_tile_size

Final best: 11940 ns  (1.0025×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1881600
@@ -33,7 +43,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -42,7 +51,7 @@

     arg_4 = inputs[4]
     xnumel = 1881600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
