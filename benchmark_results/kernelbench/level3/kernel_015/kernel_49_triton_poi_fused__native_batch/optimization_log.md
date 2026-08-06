# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_49` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9440 ns — **Best latency**: 9440 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_49`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_49_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6370,1,1), Waves/SM: 3.02
- **Registers/thread**: 32
- **Duration**: 9440 ns
- **SM Throughput**: 59.59 % | **Memory**: 21.61 % | **DRAM**: 20.10 %
- **Achieved Occupancy**: 77.84 %
- **IPC Active**: 2.98 inst/cycle | **No Eligible**: 24.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave_and_improve_cache_reuse | 0.744× | 0.648× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.000× | 0.408× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.736× | 0.653× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9440 ns | 14560 ns | +5120 ns (+54.2%) | 23140 ns | +13700 ns (+145.1%) | 14460 ns | +5020 ns (+53.2%) |
| SM Throughput (%) | 59.59 % | 63.57 % | +3.98 pp | 73.08 % | +13.49 pp | 63.34 % | +3.75 pp |
| Memory Throughput (%) | 21.61 % | 17.87 % | -3.74 pp | 18.54 % | -3.07 pp | 17.78 % | -3.83 pp |
| DRAM Throughput (%) | 20.10 % | 13.03 % | -7.07 pp | 8.19 % | -11.91 pp | 13.10 % | -7.00 pp |
| Achieved Occupancy (%) | 77.84 % | 63.94 % | -13.90 pp | 44.00 % | -33.84 pp | 63.85 % | -13.99 pp |
| Registers/Thread | 32.00 register/thread | 38.00 register/thread | +6 | 64.00 register/thread | +32 | 38.00 register/thread | +6 |
| Waves/SM | 3.02 | 4.02 | +1.00 | 6.03 | +3.01 | 4.02 | +1.00 |
| No Eligible (%) | 24.85 % | 25.77 % | +0.92 pp | 19.03 % | -5.82 pp | 25.60 % | +0.75 pp |
| IPC Active | 2.98 inst/cycle | 2.94 inst/cycle | -0.04 | 3.23 inst/cycle | +0.25 | 2.95 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_for_vectorized_loads_and_eliminate_partial_wave` (0.653×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) to a pointwise kernel that was already register-limited (32 registers/thread, Block Limit Registers=16) caused the compiler to allocate more registers per thread (32→38), reducing Block Limit Registers from 16 to 12 and dropping theoretical occupancy from 100% to 75% and achieved occupancy from 77.8% to 63.8%. The larger tile size also caused a 60% increase in total executed instructions (4.46M→7.17M) and a 53% increase in kernel duration (9.44→14.46 µs), completely negating the partial-wave elimination benefit. When a kernel is already near its register budget limit, increasing XBLOCK forces the compiler to keep more live values simultaneously, increasing register pressure and reducing occupancy more than the partial-wave savings can recover.

> **Applicable when**: Kernel has low register pressure (<<32 registers/thread), Block Limit Registers is not the binding occupancy constraint, and the partial wave is a significant fraction of runtime (>20%).

> **Avoid when**: Kernel is already at or near register budget limit (registers/thread ≥ 28-32 with Block Limit Registers already constraining occupancy). Increasing XBLOCK in this regime forces more live values, raises register count, reduces Block Limit Registers, and drops achieved occupancy more than the partial-wave elimination saves. Also avoid when the kernel has many conditional/predicated loads that scale instruction count with XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_per_thread_to_restore_occupancy | 0.998× | 0.967× | ✓ |  |
| B | remove_autotune_to_restore_register_budget_and_occupancy | 1.000× | 0.964× | ✓ |  |
| C | reduce_register_pressure_to_restore_occupancy | 0.986× | 0.949× | ✓ |  |
| D | add_triton_autotune | 0.981× | 0.955× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_per... | Δ | remove_autotune_t... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9440 ns | 9760 ns | +320 ns (+3.4%) | 9790 ns | +350 ns (+3.7%) | 9950 ns | +510 ns (+5.4%) | 9890 ns | +450 ns (+4.8%) |
| SM Throughput (%) | 59.59 % | 58.40 % | -1.19 pp | 59.60 % | +0.01 pp | 58.28 % | -1.31 pp | 59.62 % | +0.03 pp |
| Memory Throughput (%) | 21.61 % | 21.18 % | -0.43 pp | 21.61 % | +0.00 pp | 21.13 % | -0.48 pp | 21.62 % | +0.01 pp |
| DRAM Throughput (%) | 20.10 % | 19.39 % | -0.71 pp | 19.39 % | -0.71 pp | 19.11 % | -0.99 pp | 19.14 % | -0.96 pp |
| Achieved Occupancy (%) | 77.84 % | 78.61 % | +0.77 pp | 78.63 % | +0.79 pp | 78.74 % | +0.90 pp | 76.44 % | -1.40 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 |
| No Eligible (%) | 24.85 % | 23.24 % | -1.61 pp | 24.10 % | -0.75 pp | 23.99 % | -0.86 pp | 23.16 % | -1.69 pp |
| IPC Active | 2.98 inst/cycle | 3.01 inst/cycle | +0.03 | 3.02 inst/cycle | +0.04 | 2.93 inst/cycle | -0.05 | 2.82 inst/cycle | -0.16 |

---

## Cumulative Progress

```
Baseline:  9440 ns  (1.000×)

Final best: 9440 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
     xnumel = 815360
@@ -16,21 +25,21 @@

     x2 = xindex // 5824
     x3 = xindex // 416
     x4 = xindex
-    tmp45 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
-    tmp47 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
-    tmp56 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
-    tmp58 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
+    tmp45 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp47 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp56 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp58 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 256, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,26 +49,26 @@

     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 320, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 352, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 384, tl.int64)
     tmp33 = tmp0 < tmp32
     tmp34 = tmp31 & tmp33
-    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), mask=tmp34 & xmask, eviction_policy='evict_last', other=0.0)
     tmp36 = tmp0 >= tmp32
     tmp37 = tl.full([1], 416, tl.int64)
     tmp38 = tmp0 < tmp37
-    tmp39 = tl.load(in_ptr5 + (32*x3 + ((-384) + x0)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp39 = tl.load(in_ptr5 + (32*x3 + ((-384) + x0)), mask=tmp36 & xmask, eviction_policy='evict_last', other=0.0)
     tmp40 = tl.where(tmp34, tmp35, tmp39)
     tmp41 = tl.where(tmp29, tmp30, tmp40)
     tmp42 = tl.where(tmp24, tmp25, tmp41)
@@ -78,9 +87,7 @@

     tmp59 = tmp57 + tmp58
     tmp60 = tl.full([1], 0, tl.int32)
     tmp61 = tl.maximum(tmp60, tmp59)
-    tl.store(in_out_ptr0 + (x4), tmp61, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp61, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -96,7 +103,7 @@

     arg_10 = inputs[10]
     xnumel = 815360
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_49[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 815360, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_49[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 815360)
     return arg_10
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
