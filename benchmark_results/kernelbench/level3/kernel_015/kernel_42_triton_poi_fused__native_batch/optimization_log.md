# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21860 ns — **Best latency**: 21860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_42_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 18
- **Duration**: 21860 ns
- **SM Throughput**: 54.83 % | **Memory**: 23.16 % | **DRAM**: 15.51 %
- **Achieved Occupancy**: 57.63 %
- **IPC Active**: 2.46 inst/cycle | **No Eligible**: 38.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size | 0.991× | 0.984× | ✓ |  |
| B | autotune_xblock_num_warps_for_latency_hiding | 0.999× | 0.380× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads | 0.864× | 0.381× | ✓ |  |
| D | increase_xblock_with_autotune_for_occupancy_and_latency_hiding | 0.683× | 0.141× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_n... | Δ | increase_xblock_w... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 22210 ns | +350 ns (+1.6%) | 57540 ns | +35680 ns (+163.2%) | 57380 ns | +35520 ns (+162.5%) | 154780 ns | +132920 ns (+608.1%) |
| SM Throughput (%) | 54.83 % | 61.98 % | +7.15 pp | 84.97 % | +30.14 pp | 85.17 % | +30.34 pp | 79.77 % | +24.94 pp |
| Memory Throughput (%) | 23.16 % | 22.89 % | -0.27 pp | 19.41 % | -3.75 pp | 19.43 % | -3.73 pp | 21.42 % | -1.74 pp |
| DRAM Throughput (%) | 15.51 % | 15.22 % | -0.29 pp | 5.86 % | -9.65 pp | 5.90 % | -9.61 pp | 2.18 % | -13.33 pp |
| Achieved Occupancy (%) | 57.63 % | 59.48 % | +1.85 pp | 87.07 % | +29.44 pp | 86.93 % | +29.30 pp | 76.97 % | +19.34 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 26.00 register/thread | +8 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 29.70 | +14.85 | 29.70 | +14.85 | 118.79 | +103.94 |
| No Eligible (%) | 38.62 % | 30.03 % | -8.59 pp | 11.60 % | -27.02 pp | 11.68 % | -26.94 pp | 19.00 % | -19.62 pp |
| IPC Active | 2.46 inst/cycle | 2.79 inst/cycle | +0.33 | 3.53 inst/cycle | +1.07 | 3.53 inst/cycle | +1.07 | 3.24 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_num_warps_for_latency_hiding` (0.380×)

Adding @triton.autotune with larger XBLOCK configs selected a configuration (XBLOCK=256) that increased register pressure from 18 to 32 registers/thread, which reduced the number of blocks that can co-reside per SM and caused the executed instruction count to balloon from ~9.4M to ~38.5M (4x increase). Although occupancy improved from 57.6% to 87.1% and compute SM throughput rose from 54.8% to 85.0%, the massive instruction count increase dominated, causing a 2.6x slowdown in wall time (21.86 µs → 57.54 µs). The critical implementation flaw was that the grid was left hardcoded to `triton.cdiv(xnumel, 128)` instead of being updated to use `meta['XBLOCK']`, meaning the autotune selected a larger XBLOCK but the grid was still sized for XBLOCK=128, causing each thread to process far more elements than intended and executing redundant work.

> **Applicable when**: Kernel has low achieved occupancy (<60%) due to too few warps per block, CPIStall rule fires with L1TEX stall fraction >40%, eligible warps per scheduler <2.0, and the kernel body is simple enough that larger tiles don't cause register spilling.

> **Avoid when**: When adding autotune, the call-site grid must be updated to a lambda using meta['XBLOCK']. If the grid remains hardcoded to the original XBLOCK, autotune will select a larger block size while the grid stays sized for the smaller block, causing each thread to process multiple elements and exploding instruction count. Also avoid when register count per thread is already near the per-SM register file limit, as larger XBLOCK configs may increase register pressure and reduce actual occupancy.

#### ❌ Plan C: `increase_xblock_with_autotune_for_vectorized_loads` (0.381×)

Adding @triton.autotune with larger XBLOCK configs (512/1024) caused a 4x increase in executed instructions (9.4M → 38.5M) and a 2.6x increase in elapsed cycles (32K → 86K), resulting in a 0.38x regression. The autotune selected XBLOCK=256 (not 512/1024 as intended), and the grid was left hardcoded at cdiv(xnumel, 128) rather than updated to a lambda using meta['XBLOCK'], meaning the kernel was launched with the wrong grid size — each thread processed more elements than intended, inflating instruction count. Additionally, doubling registers per thread (18→32) halved the block limit from registers (21→8 blocks/SM), capping occupancy gains despite the higher achieved occupancy metric.

> **Applicable when**: Never apply this exact pattern — the grid was not updated correctly when adding autotune.

> **Avoid when**: Do NOT add @triton.autotune with variable XBLOCK without simultaneously updating the launch grid to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Mismatched grid and XBLOCK causes each thread to process multiple elements, exploding instruction count. Also avoid increasing XBLOCK when registers per thread will increase significantly (e.g., 18→32), as this reduces block-limit-registers and can negate occupancy gains.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_increase_num_warps_for_latency_hiding | 0.998× | 0.991× | ✓ |  |
| B | reduce_non_fused_fp32_instructions_via_fma_rewrite | 0.865× | 0.380× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_instruction_explosion | 0.991× | 0.379× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_work | 0.683× | 0.142× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | reduce_non_fused_... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21860 ns | 22050 ns | +190 ns (+0.9%) | 57540 ns | +35680 ns (+163.2%) | 57700 ns | +35840 ns (+164.0%) | 154400 ns | +132540 ns (+606.3%) |
| SM Throughput (%) | 54.83 % | 62.59 % | +7.76 pp | 85.10 % | +30.27 pp | 85.09 % | +30.26 pp | 79.66 % | +24.83 pp |
| Memory Throughput (%) | 23.16 % | 23.15 % | -0.01 pp | 19.43 % | -3.73 pp | 19.42 % | -3.74 pp | 21.39 % | -1.77 pp |
| DRAM Throughput (%) | 15.51 % | 15.42 % | -0.09 pp | 5.86 % | -9.65 pp | 5.89 % | -9.62 pp | 2.19 % | -13.32 pp |
| Achieved Occupancy (%) | 57.63 % | 59.76 % | +2.13 pp | 87.12 % | +29.49 pp | 87.03 % | +29.40 pp | 76.98 % | +19.35 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 26.00 register/thread | +8 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 29.70 | +14.85 | 29.70 | +14.85 | 118.79 | +103.94 |
| No Eligible (%) | 38.62 % | 28.88 % | -9.74 pp | 11.56 % | -27.06 pp | 11.72 % | -26.90 pp | 19.02 % | -19.60 pp |
| IPC Active | 2.46 inst/cycle | 2.80 inst/cycle | +0.34 | 3.54 inst/cycle | +1.08 | 3.53 inst/cycle | +1.07 | 3.24 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan B: `reduce_non_fused_fp32_instructions_via_fma_rewrite` (0.380×)

The intended optimization was to eliminate int32 intermediates and use tl.fma() to reduce non-fused FP32 instructions, but the actual regression was caused by simultaneously changing the @triton.autotune configs to larger XBLOCK values (1024, 512) without updating the launch grid from a static `cdiv(xnumel, 128)` to a lambda using `meta['XBLOCK']`. This caused the same instruction explosion seen in iteration 1: executed instructions ballooned from ~10.8M to ~38.5M (3.6x) and elapsed cycles jumped from 33K to 86K (2.6x), producing a 0.38x regression. The FMA arithmetic rewrite itself (removing int32 temporaries, using 1.0/tmp6, tl.fma, tl.maximum(0.0, ...)) is likely correct and harmless, but its effect is completely masked by the grid/autotune mismatch bug.

> **Applicable when**: Never apply this combined pattern. The FMA arithmetic rewrite alone (removing int32 intermediates, using tl.fma) may be valid when FPInstructions NCU rule fires with high non-fused FP32 instruction count and the kernel is compute-bound (SM throughput >80%). However, it must be applied WITHOUT simultaneously changing autotune configs unless the grid is also updated.

> **Avoid when**: Do NOT combine an arithmetic rewrite with autotune config changes in the same iteration unless the launch grid is simultaneously updated to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. The autotune config change is the dominant regression source and will completely mask any benefit from the FMA rewrite. Also do not apply when registers per thread will increase significantly (20→32), as this reduces block-limit-registers and caps occupancy.

---

## Cumulative Progress

```
Baseline:  21860 ns  (1.000×)

Final best: 21860 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=16),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 1024}, num_warps=32),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4014080
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 512)
-    tmp0 = tl.load(in_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -31,9 +43,7 @@

     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +54,7 @@

     arg_5 = inputs[5]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4014080)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
