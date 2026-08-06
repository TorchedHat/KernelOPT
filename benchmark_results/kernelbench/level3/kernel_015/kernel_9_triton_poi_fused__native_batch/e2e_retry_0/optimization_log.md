# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 26780 ns — **Best latency**: 62240 ns (**0.4303×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (39200,1,1), Waves/SM: 18.56
- **Registers/thread**: 28
- **Duration**: 26780 ns
- **SM Throughput**: 65.26 % | **Memory**: 32.61 % | **DRAM**: 17.25 %
- **Achieved Occupancy**: 54.69 %
- **IPC Active**: 2.85 inst/cycle | **No Eligible**: 28.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.904× | 0.825× | ✓ |  |
| B | autotune_xblock_num_warps_num_stages | 1.005× | 0.430× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 26780 ns | 32480 ns | +5700 ns (+21.3%) | 62240 ns | +35460 ns (+132.4%) |
| SM Throughput (%) | 65.26 % | 73.73 % | +8.47 pp | 76.42 % | +11.16 pp |
| Memory Throughput (%) | 32.61 % | 35.56 % | +2.95 pp | 29.95 % | -2.66 pp |
| DRAM Throughput (%) | 17.25 % | 14.09 % | -3.16 pp | 7.33 % | -9.92 pp |
| Achieved Occupancy (%) | 54.69 % | 85.01 % | +30.32 pp | 82.50 % | +27.81 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 18.56 | 18.56 | +0.00 | 37.12 | +18.56 |
| No Eligible (%) | 28.04 % | 21.09 % | -6.95 pp | 20.70 % | -7.34 pp |
| IPC Active | 2.85 inst/cycle | 3.16 inst/cycle | +0.31 | 3.16 inst/cycle | +0.31 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.825×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) increased achieved occupancy from 54.7% to 85.0% and reduced the L1TEX scoreboard stall fraction, but the kernel regressed by ~21% (26.78 µs → 32.48 µs). The autotune selected a config that increased registers/thread from 28 to 32, which reduced the block limit from registers and caused Warp Cycles Per Issued Instruction to worsen from 11.8 to 17.1 cycles — the dominant stall shifted from L1TEX latency to 'not selected' scheduler stalls, indicating the kernel became over-subscribed with warps competing for issue slots. Additionally, executed instructions ballooned from 13.6M to 18.8M (38% more), suggesting the autotune selected a config that introduced extra work (likely loop unrolling overhead or index recomputation at larger tile sizes). Higher occupancy alone does not guarantee speedup when the per-instruction cost increases and the memory pipe becomes less saturated (Mem Pipes Busy dropped from 54.2% to 44.5%).

> **Applicable when**: Kernel has very low achieved occupancy (<40%) AND the dominant stall is L1TEX scoreboard AND registers/thread are well below the block register limit AND instruction count is stable across tile sizes.

> **Avoid when**: Registers/thread are near the block register limit (e.g., 28/32 with block_limit_registers=16), because larger XBLOCK configs selected by autotune may push registers/thread higher (28→32), reducing the block limit and causing scheduler 'not selected' stalls to dominate. Also avoid when the kernel's instruction count scales super-linearly with XBLOCK (e.g., due to unrolling), as the extra instructions can outweigh latency-hiding benefits. Do not apply when achieved occupancy is already moderate (>50%) and the primary stall is not L1TEX latency.

#### ❌ Plan B: `autotune_xblock_num_warps_num_stages` (0.430×)

Adding @triton.autotune to explore larger XBLOCK values (128→1024) and more warps increased achieved occupancy from 54.7% to 82.5%, but the selected config (XBLOCK=256, block_size=256) caused a 2.3x increase in total executed instructions (13.6M → 37.4M) and a 2x increase in kernel duration (26.8 µs → 62.2 µs). The autotune overhead itself (benchmark runs across 8 configs) likely dominated the measured latency, and the winning config was not the largest block size — indicating the grid was not updated to use `lambda meta:` and remained fixed at `cdiv(xnumel, 128)`, causing each thread to process 2x the work with XBLOCK=256 while the grid stayed the same size. When the grid is not updated to match the autotuned XBLOCK, each CTA processes more elements than intended, inflating instruction count and runtime.

> **Applicable when**: Never apply this pattern — it is an implementation flaw. The correct pattern requires grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']),)

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK unless the grid lambda is simultaneously updated to use meta['XBLOCK']. Keeping a hardcoded grid size while autotuning XBLOCK causes each CTA to process more elements than the tile size, multiplying instruction count and runtime.

---

## Cumulative Progress

```
Baseline:  26780 ns  (1.000×)

Final best: 62240 ns  (0.4303×)
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
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=2, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
     xnumel = 5017600
@@ -52,7 +66,6 @@

     tmp39 = tl.maximum(tmp38, tmp37)
     tl.store(in_out_ptr0 + (x2), tmp39, mask=xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -65,7 +78,7 @@

     arg_8 = inputs[8]
     xnumel = 5017600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600)
     return arg_8
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
