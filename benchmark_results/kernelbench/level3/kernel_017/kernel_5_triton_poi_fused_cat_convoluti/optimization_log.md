# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2440000 ns — **Best latency**: 646690 ns (**3.7731×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`coalesce_inner_load_by_swapping_xblock_yblock_roles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_017/kernel_5_triton_poi_fused_cat_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 32
- **Duration**: 2440000 ns
- **SM Throughput**: 30.43 % | **Memory**: 73.85 % | **DRAM**: 71.38 %
- **Achieved Occupancy**: 95.81 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 69.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_swap_tile_dimensions | 1.000× | 1.000× | ✓ |  |
| B | software_pipelining_via_autotune | 1.481× | 1.000× | ✓ | **✓ best** |
| C | increase_tile_sizes_with_autotune | 0.288× | 0.618× | ✓ |  |
| D | autotune_tile_sizes_for_l2_pressure_reduction | 0.422× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | software_pipelini... | Δ | increase_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2440000 ns | 2440000 ns | +0 ns (+0.0%) | 2440000 ns | +0 ns (+0.0%) | 3950000 ns | +1510000 ns (+61.9%) | 2450000 ns | +10000 ns (+0.4%) |
| SM Throughput (%) | 30.43 % | 30.45 % | +0.02 pp | 30.45 % | +0.02 pp | 51.83 % | +21.40 pp | 30.71 % | +0.28 pp |
| Memory Throughput (%) | 73.85 % | 73.82 % | -0.03 pp | 73.81 % | -0.04 pp | 82.31 % | +8.46 pp | 73.81 % | -0.04 pp |
| DRAM Throughput (%) | 71.38 % | 71.36 % | -0.02 pp | 71.36 % | -0.02 pp | 44.15 % | -27.23 pp | 71.44 % | +0.06 pp |
| Achieved Occupancy (%) | 95.81 % | 95.79 % | -0.02 pp | 95.79 % | -0.02 pp | 29.21 % | -66.60 pp | 95.81 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 81.00 register/thread | +49 | 32.00 register/thread | +0 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 1588.75 | +1092.27 | 496.48 | +0.00 |
| No Eligible (%) | 69.95 % | 69.99 % | +0.04 pp | 69.97 % | +0.02 pp | 48.84 % | -21.11 pp | 70.04 % | +0.09 pp |
| IPC Active | 1.20 inst/cycle | 1.20 inst/cycle | +0.00 | 1.21 inst/cycle | +0.01 | 2.05 inst/cycle | +0.85 | 1.20 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_l2_pressure_reduction | 0.342× | 0.777× | ✓ |  |
| B | streaming_eviction_policy_for_non_reused_loads | 0.732× | 0.663× | ✓ |  |
| C | streaming_cache_eviction_policy_for_l1_bypass | 0.288× | 0.667× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_cap | 1.657× | 3.775× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | streaming_evictio... | Δ | streaming_cache_e... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2440000 ns | 3140000 ns | +700000 ns (+28.7%) | 3680000 ns | +1240000 ns (+50.8%) | 3660000 ns | +1220000 ns (+50.0%) | 646370 ns | -1793630 ns (-73.5%) |
| SM Throughput (%) | 30.43 % | 26.16 % | -4.27 pp | 21.40 % | -9.03 pp | 20.95 % | -9.48 pp | 45.79 % | +15.36 pp |
| Memory Throughput (%) | 73.85 % | 61.06 % | -12.79 pp | 61.88 % | -11.97 pp | 62.13 % | -11.72 pp | 69.73 % | -4.12 pp |
| DRAM Throughput (%) | 71.38 % | 58.67 % | -12.71 pp | 61.88 % | -9.50 pp | 62.13 % | -9.25 pp | 67.67 % | -3.71 pp |
| Achieved Occupancy (%) | 95.81 % | 96.27 % | +0.46 pp | 96.77 % | +0.96 pp | 96.76 % | +0.95 pp | 54.44 % | -41.37 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 496.48 | +0.00 | 248.24 | -248.24 |
| No Eligible (%) | 69.95 % | 73.30 % | +3.35 pp | 78.63 % | +8.68 pp | 78.58 % | +8.63 pp | 56.90 % | -13.05 pp |
| IPC Active | 1.20 inst/cycle | 1.09 inst/cycle | -0.11 | 0.85 inst/cycle | -0.35 | 0.86 inst/cycle | -0.34 | 1.72 inst/cycle | +0.52 |


### Key Learnings

#### ❌ Plan A: `streaming_eviction_policy_for_l2_pressure_reduction` (0.777×)

Adding eviction_policy='evict_first' to large streaming tensor loads and stores in a Triton kernel reduced L2 Cache Throughput from 96.33% to 75.58% and DRAM Throughput from 71.36% to 58.67%, confirming the hint was received by the hardware. However, total executed instructions increased by ~10% (578M → 637M) and elapsed cycles grew from 3.66M to 4.71M, resulting in a 0.78x regression (2.44ms → 3.14ms). The warp stall cycles per issued instruction barely changed (50.84 → 49.3), indicating the underlying non-coalesced access pattern remained the bottleneck; the eviction hint reduced L2 pressure but did not improve memory-level parallelism or coalescing, and the instruction overhead of the policy change outweighed any cache-pollution reduction. When the root cause is non-coalesced access (stride >> cache-line width), eviction policy hints cannot compensate — only access pattern restructuring can fix the stall.

> **Applicable when**: Never applicable as described; this is a negative example. The evict_first hint on streaming loads is not beneficial when the kernel's primary bottleneck is non-coalesced memory access (warp cycles per issued instruction ~50, L1 hit rate <2%) rather than L2 cache pollution from reused data.

> **Avoid when**: Do not apply eviction_policy='evict_first' to streaming loads when: (1) warp stall cycles remain high (~50 cycles/instruction) due to non-coalesced access patterns, (2) L1/TEX hit rate is already near 0% (no L1 reuse to protect), and (3) the kernel is already memory-bandwidth bound at the DRAM level. Reducing L2 throughput via eviction hints does not improve coalescing efficiency and may increase instruction count overhead. The fix for non-coalesced access is access pattern restructuring (transposing data layout, changing tile dimensions to match stride), not cache policy hints.

---

## Iteration 3

**Best so far**: 3.7749× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_inner_load_by_swapping_xblock_yblock_roles | 3.760× | 3.773× | ✓ | **✓ best** |
| B | autotune_smaller_xblock_with_maxnreg_for_latency_hiding | 3.293× | 3.649× | ✓ |  |
| C | cap_registers_with_maxnreg_and_smaller_xblock_configs | 3.051× | 3.069× | ✓ |  |
| D | maxnreg_cap_with_smaller_block_autotune_for_latency_hiding | 3.053× | 3.026× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_inner_lo... | Δ | autotune_smaller_... | Δ | cap_registers_wit... | Δ | maxnreg_cap_with_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2440000 ns | 646690 ns | -1793310 ns (-73.5%) | 668740 ns | -1771260 ns (-72.6%) | 794980 ns | -1645020 ns (-67.4%) | 806460 ns | -1633540 ns (-66.9%) |
| SM Throughput (%) | 30.43 % | 47.06 % | +16.63 pp | 46.91 % | +16.48 pp | 37.09 % | +6.66 pp | 37.15 % | +6.72 pp |
| Memory Throughput (%) | 73.85 % | 69.70 % | -4.15 pp | 67.42 % | -6.43 pp | 59.39 % | -14.46 pp | 58.48 % | -15.37 pp |
| DRAM Throughput (%) | 71.38 % | 67.06 % | -4.32 pp | 64.75 % | -6.63 pp | 58.30 % | -13.08 pp | 58.12 % | -13.26 pp |
| Achieved Occupancy (%) | 95.81 % | 52.10 % | -43.71 pp | 52.43 % | -43.38 pp | 88.79 % | -7.02 pp | 88.60 % | -7.21 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 248.24 | -248.24 | 248.24 | -248.24 | 248.24 | -248.24 |
| No Eligible (%) | 69.95 % | 56.28 % | -13.67 pp | 52.79 % | -17.16 pp | 63.43 % | -6.52 pp | 63.42 % | -6.53 pp |
| IPC Active | 1.20 inst/cycle | 1.74 inst/cycle | +0.54 | 1.90 inst/cycle | +0.70 | 1.47 inst/cycle | +0.27 | 1.47 inst/cycle | +0.27 |

---

## Iteration 4

**Best so far**: 3.7749× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_arithmetic_restructure | 3.758× | 3.775× | ✓ |  |
| B | fused_fp32_bias_relu_via_explicit_fma_restructure | 3.293× | 3.760× | ✓ |  |
| C | increase_num_warps_for_l1tex_latency_hiding | 3.052× | 3.069× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_ar... | Δ | fused_fp32_bias_r... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2440000 ns | 646340 ns | -1793660 ns (-73.5%) | 648960 ns | -1791040 ns (-73.4%) | 795040 ns | -1644960 ns (-67.4%) |
| SM Throughput (%) | 30.43 % | 46.92 % | +16.49 pp | 47.19 % | +16.76 pp | 36.73 % | +6.30 pp |
| Memory Throughput (%) | 73.85 % | 69.76 % | -4.09 pp | 69.48 % | -4.37 pp | 59.30 % | -14.55 pp |
| DRAM Throughput (%) | 71.38 % | 67.11 % | -4.27 pp | 66.74 % | -4.64 pp | 58.24 % | -13.14 pp |
| Achieved Occupancy (%) | 95.81 % | 52.28 % | -43.53 pp | 52.19 % | -43.62 pp | 88.13 % | -7.68 pp |
| Registers/Thread | 32.00 register/thread | 26.00 register/thread | -6 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 248.24 | -248.24 | 248.24 | -248.24 |
| No Eligible (%) | 69.95 % | 56.32 % | -13.63 pp | 52.78 % | -17.17 pp | 62.93 % | -7.02 pp |
| IPC Active | 1.20 inst/cycle | 1.74 inst/cycle | +0.54 | 1.89 inst/cycle | +0.69 | 1.47 inst/cycle | +0.27 |


### Key Learnings

#### ❌ Plan A: `fma_fusion_via_arithmetic_restructure` (3.775×)

Replacing tl.full([1,1], 0, tl.int32) integer-zero ReLU clamps with float literal 0.0 and using libdevice.fma for bias-add (motivated by NCU FPInstructions rule reporting 0 fused / 8.4M non-fused FP32 instructions) produced zero measurable improvement: elapsed cycles dropped by only 0.26% (965546 → 963048), executed instruction count was unchanged (220.2M), and all scheduler/occupancy/memory metrics were statistically identical before and after. The kernel is deeply L2/DRAM-bound (L2 throughput 95.6%, DRAM 67%), and the FP32 instruction path is not on the critical latency path — the bottleneck is memory scoreboard stalls (CPIStall L1TEX at 70% of stall cycles), not ALU throughput. When a kernel is memory-latency bound with warp stall cycles dominated by L1TEX scoreboard waits, FMA fusion cannot improve performance because the ALU pipeline is not the bottleneck.

> **Applicable when**: Never applicable as a standalone optimization when the kernel is L2/DRAM-bound with CPIStall L1TEX dominating stall cycles (>60% of warp stall time). The NCU FPInstructions rule may fire, but if executed instruction count does not change and memory metrics are unchanged, the FMA fusion had no effect on the PTX/SASS emitted.

> **Avoid when**: Kernel is memory-latency bound (L2 Cache Throughput >90%, DRAM Throughput >60%, CPIStall L1TEX >60% of stall cycles). In this regime, ALU instruction count reduction via FMA fusion does not reduce elapsed cycles because the critical path is memory scoreboard waits, not ALU throughput. Also do not apply when the NCU FPInstructions rule reports a potential gain but the kernel's compute SM throughput is already well below the memory throughput (Compute SM Throughput <50% while Memory Throughput >65%) — the compute pipeline is not the bottleneck.

#### ✅ Plan C: `increase_num_warps_for_l1tex_latency_hiding` (3.069×)

Expanding the Triton autotune config set with high num_warps (8–16) and num_stages (3–4) variants caused the autotuner to select a configuration that increased active warps per scheduler from 8.35 to 14.3 (more in-flight warps) but simultaneously worsened warp cycles per issued instruction from 19.1 to 38.59 — doubling the stall depth. Despite the higher occupancy (52.1% → 88.1%), the selected config increased register pressure (26 → 30 regs/thread) and the no-eligible-warp rate rose from 56.3% to 62.9%, indicating the additional warps were not hiding latency but instead competing for memory bandwidth and increasing contention. The kernel duration increased from 647 µs to 795 µs, yielding a net regression on this kernel in isolation; the reported 3.07x speedup reflects end-to-end model improvement from other kernels affected by the autotuning, not this specific kernel's improvement.

> **Applicable when**: Never applicable as described for this kernel; this is a cautionary example. Adding high num_warps configs to autotune is risky when the baseline kernel already has low warp cycles per issued instruction (<25) and moderate occupancy (>50%), as the autotuner may select a config that increases register pressure and stall depth simultaneously.

> **Avoid when**: Kernel already has warp cycles per issued instruction below 25 and achieved occupancy above 50% with maxnreg capping in place. In this regime, adding more warps via higher num_warps configs increases active warps per scheduler but does not reduce the no-eligible-warp rate — the bottleneck is memory bandwidth contention, not latency depth. Do not apply when registers-per-thread are already near the maxnreg cap (26–30 regs with maxnreg=64), as adding num_stages=4 can push register usage higher and worsen per-warp stall cycles.

---

## Iteration 5

**Best so far**: 3.7749× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | deeper_software_pipeline_to_hide_l1tex_latency | 3.292× | 3.636× | ✓ |  |


### Metric Comparison

| Metric | Baseline | deeper_software_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2440000 ns | 671140 ns | -1768860 ns (-72.5%) |
| SM Throughput (%) | 30.43 % | 46.92 % | +16.49 pp |
| Memory Throughput (%) | 73.85 % | 67.16 % | -6.69 pp |
| DRAM Throughput (%) | 71.38 % | 64.52 % | -6.86 pp |
| Achieved Occupancy (%) | 95.81 % | 52.34 % | -43.47 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 |
| Waves/SM | 496.48 | 248.24 | -248.24 |
| No Eligible (%) | 69.95 % | 52.87 % | -17.08 pp |
| IPC Active | 1.20 inst/cycle | 1.90 inst/cycle | +0.70 |

---

## Cumulative Progress

```
Baseline:  2440000 ns  (1.000×)
Iter 2     646370 ns  (3.775×)  — reduce_register_pressure_via_maxnreg_cap

Final best: 646690 ns  (3.7731×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,19 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8,  num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8,  num_stages=2, maxnreg=64),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_cat_convolution_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 65536
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
     ymask = yoffset + tl.arange(0, YBLOCK) < ynumel
-    ymask = ymask[:, None]
+    ymask = ymask[None, :]
     xoffset = tl.program_id(0) * XBLOCK
     _xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
-    xindex = _xrange[None, :]
+    xindex = _xrange[:, None]
     xmask = xoffset + tl.arange(0, XBLOCK) < xnumel
-    xmask = xmask[None, :]
+    xmask = xmask[:, None]
     y0 = (yindex % 128)
     x2 = xindex
     y1 = yindex // 128
@@ -28,23 +42,18 @@

     tmp3 = tl.full([1, 1], 64, tl.int64)
     tmp4 = tmp0 < tmp3
     tmp5 = tl.load(in_ptr0 + (64*x2 + 4194304*y1 + (y0)), tmp4 & ymask & xmask, other=0.0)
-    tmp6 = tl.full([1, 1], 0, tl.int32)
-    tmp7 = tl.maximum(tmp6, tmp5)
-    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
-    tmp9 = tl.where(tmp4, tmp7, tmp8)
+    tmp7 = tl.maximum(0.0, tmp5)
+    tmp9 = tl.where(tmp4, tmp7, 0.0)
     tmp10 = tmp0 >= tmp3
     tmp11 = tl.full([1, 1], 128, tl.int64)
     tmp12 = tmp0 < tmp11
     tmp13 = tl.load(in_ptr1 + (64*x2 + 4194304*y1 + ((-64) + y0)), tmp10 & ymask & xmask, other=0.0)
-    tmp14 = tl.load(in_ptr2 + (tl.broadcast_to((-64) + y0, [YBLOCK, XBLOCK])), tmp10 & ymask, eviction_policy='evict_last', other=0.0)
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1, 1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
-    tmp19 = tl.where(tmp10, tmp17, tmp18)
+    tmp14 = tl.load(in_ptr2 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp10 & ymask, eviction_policy='evict_last', other=0.0)
+    tmp15 = libdevice.fma(tmp13, tl.full(tmp13.shape, 1.0, tl.float32), tmp14)
+    tmp17 = tl.maximum(0.0, tmp15)
+    tmp19 = tl.where(tmp10, tmp17, 0.0)
     tmp20 = tl.where(tmp4, tmp9, tmp19)
     tl.store(out_ptr0 + (x2 + 65536*y3), tmp20, ymask & xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -54,7 +63,7 @@

     ynumel = 16384
     xnumel = 65536
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_cat_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, 16384, 65536, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_cat_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, 16384, 65536)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
