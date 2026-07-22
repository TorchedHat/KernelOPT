# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_gelu_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 359520 ns — **Best latency**: 359520 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_gelu_relu_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_041/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524288,1,1), Waves/SM: 248.24
- **Registers/thread**: 20
- **Duration**: 359520 ns
- **SM Throughput**: 81.03 % | **Memory**: 28.89 % | **DRAM**: 28.89 %
- **Achieved Occupancy**: 81.61 %
- **IPC Active**: 3.26 inst/cycle | **No Eligible**: 18.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_compute_efficiency | 0.647× | 0.104× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_fma_and_occupancy | 0.922× | 0.922× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_reduced_launch_overhead | 0.398× | 0.395× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 359520 ns | 3440000 ns | +3080480 ns (+856.8%) | 390020 ns | +30500 ns (+8.5%) | 910270 ns | +550750 ns (+153.2%) |
| SM Throughput (%) | 81.03 % | 88.93 % | +7.90 pp | 86.32 % | +5.29 pp | 89.28 % | +8.25 pp |
| Memory Throughput (%) | 28.89 % | 10.42 % | -18.47 pp | 26.63 % | -2.26 pp | 15.44 % | -13.45 pp |
| DRAM Throughput (%) | 28.89 % | 3.01 % | -25.88 pp | 26.63 % | -2.26 pp | 11.41 % | -17.48 pp |
| Achieved Occupancy (%) | 81.61 % | 46.55 % | -35.06 pp | 84.86 % | +3.25 pp | 90.84 % | +9.23 pp |
| Registers/Thread | 20.00 register/thread | 61.00 register/thread | +41 | 20.00 register/thread | +0 | 32.00 register/thread | +12 |
| Waves/SM | 248.24 | 992.97 | +744.73 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 18.31 % | 11.01 % | -7.30 pp | 13.16 % | -5.15 pp | 10.48 % | -7.83 pp |
| IPC Active | 3.26 inst/cycle | 3.56 inst/cycle | +0.30 | 3.47 inst/cycle | +0.21 | 3.58 inst/cycle | +0.32 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_compute_efficiency` (0.104×)

Adding @triton.autotune with larger XBLOCK values (256–2048) to a pointwise kernel that was already compute-bound caused registers per thread to jump from 20 to 61, collapsing theoretical occupancy from 100% to 50% and Block Limit Registers from 21 to 4. This increased waves/SM from 248 to 993 and total executed instructions by 10x, resulting in a 9.6x slowdown. When a Triton kernel's register count is already near the per-SM register file limit, increasing XBLOCK forces the compiler to allocate more registers for the wider tile's live values, causing severe occupancy collapse that far outweighs any FMA fusion benefit.

> **Applicable when**: Kernel has very high registers/thread (>40) AND low theoretical occupancy (<60%) AND is memory-bound — wider tiles can amortize memory latency without worsening occupancy further.

> **Avoid when**: Kernel already has low registers/thread (<=24) and high theoretical occupancy (>=90%); increasing XBLOCK in Triton causes the compiler to allocate significantly more registers for the wider tile, collapsing occupancy and dramatically increasing total instruction count. Also avoid when the kernel is already compute-bound with high waves/SM — wider tiles increase serialization.

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_reduced_launch_overhead` (0.528×)

Combining libdevice.rsqrt arithmetic simplification with @triton.autotune over larger XBLOCK configs (512–2048) on a compute-bound kernel caused executed instructions to balloon from 266M to 456M (1.7x increase) and duration to increase from 390 µs to 681 µs (1.75x slower). Registers per thread rose from 20 to 32, and DRAM throughput dropped from 26.6% to 15.3%, indicating the wider tiles did not improve memory efficiency. The rsqrt substitution itself reduced branch instructions (from 20.9M to 2.6M) and eliminated divergent branches, but the XBLOCK increase dominated with its instruction count inflation and register pressure increase. This confirms that on compute-bound pointwise kernels with low initial register counts, any XBLOCK increase via autotune regresses performance regardless of arithmetic simplifications applied simultaneously.

> **Applicable when**: Kernel is memory-latency-bound (low DRAM throughput <30%, high memory stall cycles), and XBLOCK is small enough that loads are not coalesced into 128-bit transactions.

> **Avoid when**: Kernel is already compute-bound (SM throughput >80%) with low initial register count (<=24 registers/thread); increasing XBLOCK via autotune will increase register pressure from 20 to 32+, inflate total instruction count by 1.7x–2.8x, and cause 1.75x–2.5x slowdown regardless of any arithmetic simplifications applied simultaneously. The XBLOCK increase effect dominates over any arithmetic savings.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_main_tensor_to_improve_l2_broadcast_caching | 0.896× | 0.894× | ✓ |  |
| B | simplify_batchnorm_arithmetic_to_reduce_nonfused_fp32_instructions | 0.534× | 0.528× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.647× | 0.642× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_main_... | Δ | simplify_batchnor... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 359520 ns | 402110 ns | +42590 ns (+11.8%) | 681020 ns | +321500 ns (+89.4%) | 559970 ns | +200450 ns (+55.8%) |
| SM Throughput (%) | 81.03 % | 87.01 % | +5.98 pp | 85.35 % | +4.32 pp | 88.33 % | +7.30 pp |
| Memory Throughput (%) | 28.89 % | 25.84 % | -3.05 pp | 20.85 % | -8.04 pp | 22.11 % | -6.78 pp |
| DRAM Throughput (%) | 28.89 % | 25.84 % | -3.05 pp | 15.25 % | -13.64 pp | 18.56 % | -10.33 pp |
| Achieved Occupancy (%) | 81.61 % | 85.84 % | +4.23 pp | 87.43 % | +5.82 pp | 88.26 % | +6.65 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 | 32.00 register/thread | +12 | 28.00 register/thread | +8 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 18.31 % | 12.54 % | -5.77 pp | 14.31 % | -4.00 pp | 11.28 % | -7.03 pp |
| IPC Active | 3.26 inst/cycle | 3.49 inst/cycle | +0.23 | 3.43 inst/cycle | +0.17 | 3.55 inst/cycle | +0.29 |


### Key Learnings

#### ❌ Plan C: `reduce_register_pressure_via_maxnreg_to_restore_occupancy` (0.642×)

Applying maxnreg=32/40 to a Triton autotune config when the baseline kernel already uses only 20 registers/thread caused registers to increase to 28 and total executed instructions to jump from 266M to 390M (+46%), increasing duration from 390µs to 560µs (0.64x). The kernel was already at 100% theoretical occupancy and 84.9% achieved occupancy before the change; adding maxnreg alongside larger XBLOCK configs (256–512) forced the compiler to spill or restructure register allocation in a way that increased instruction count without improving occupancy. When a kernel is already register-efficient and compute-bound with high occupancy, adding maxnreg constraints combined with wider XBLOCK tiles is counterproductive — the wider tiles increase total work per block and the register cap forces additional spill/reload instructions.

> **Applicable when**: maxnreg is useful when registers/thread is genuinely high (>40) AND Block Limit Registers is the binding occupancy constraint (e.g., Block Limit Registers << Block Limit Warps). The baseline kernel must have low theoretical occupancy (<75%) caused specifically by register pressure.

> **Avoid when**: Kernel already has low registers/thread (<=24) and theoretical occupancy at or near 100%. In this regime, adding maxnreg combined with wider XBLOCK autotune configs increases total instruction count (spill/reload overhead + more work per block) without improving occupancy, causing regression. Also avoid when the kernel is already compute-bound at high SM throughput (>85%) with high achieved occupancy (>80%) — there is no occupancy headroom to recover.

---

## Cumulative Progress

```
Baseline:  359520 ns  (1.000×)

Final best: 359520 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 128}, num_warps=8),
+    triton.Config({'XBLOCK': 256}, num_warps=4),
+    triton.Config({'XBLOCK': 256}, num_warps=8),
+    triton.Config({'XBLOCK': 512}, num_warps=4),
+    triton.Config({'XBLOCK': 512}, num_warps=8),
+    triton.Config({'XBLOCK': 1024}, num_warps=4),
+    triton.Config({'XBLOCK': 1024}, num_warps=8),
+], key=['xnumel'])
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_addmm_gelu_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 67108864
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 4096)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 - tmp3
     tmp6 = tl.full([1], 1e-05, tl.float32)
@@ -40,9 +51,7 @@

     tmp24 = tmp19 * tmp23
     tmp25 = tl.full([1], 0, tl.int32)
     tmp26 = tl.maximum(tmp25, tmp24)
-    tl.store(in_out_ptr0 + (x2), tmp26, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -53,7 +62,7 @@

     arg_5 = inputs[5]
     xnumel = 67108864
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_addmm_gelu_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 67108864, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_addmm_gelu_relu_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 67108864)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
