# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 174940 ns — **Best latency**: 175420 ns (**0.9973×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_2_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (73728,1,1), Waves/SM: 34.91
- **Registers/thread**: 32
- **Duration**: 174940 ns
- **SM Throughput**: 27.85 % | **Memory**: 81.19 % | **DRAM**: 67.14 %
- **Achieved Occupancy**: 94.18 %
- **IPC Active**: 1.17 inst/cycle | **No Eligible**: 71.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_staged_transpose_for_coalesced_access | 0.996× | 1.003× | ✓ |  |
| D | increase_yblock_for_bias_reuse_and_l2_pressure_relief | 1.000× | 0.997× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | increase_yblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 174940 ns | 174340 ns | -600 ns (-0.3%) | 175420 ns | +480 ns (+0.3%) |
| SM Throughput (%) | 27.85 % | 29.68 % | +1.83 pp | 27.67 % | -0.18 pp |
| Memory Throughput (%) | 81.19 % | 81.94 % | +0.75 pp | 80.95 % | -0.24 pp |
| DRAM Throughput (%) | 67.14 % | 67.36 % | +0.22 pp | 66.93 % | -0.21 pp |
| Achieved Occupancy (%) | 94.18 % | 93.85 % | -0.33 pp | 94.19 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.03 % | 69.87 % | -1.16 pp | 70.72 % | -0.31 pp |
| IPC Active | 1.17 inst/cycle | 1.17 inst/cycle | +0.00 | 1.16 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_stores_with_contiguous_hints_and_evict_first_loads | 0.781× | 0.809× | ✓ |  |
| B | eviction_policy_tuning_to_reduce_l2_saturation | 0.799× | 0.950× | ✓ |  |
| C | vectorized_stores_with_contiguous_hints_and_evict_first_loads | 0.775× | 0.792× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_stores... | Δ | eviction_policy_t... | Δ | vectorized_stores... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174940 ns | 216260 ns | +41320 ns (+23.6%) | 184160 ns | +9220 ns (+5.3%) | 220930 ns | +45990 ns (+26.3%) |
| SM Throughput (%) | 27.85 % | 23.16 % | -4.69 pp | 27.83 % | -0.02 pp | 21.12 % | -6.73 pp |
| Memory Throughput (%) | 81.19 % | 68.08 % | -13.11 pp | 76.76 % | -4.43 pp | 66.89 % | -14.30 pp |
| DRAM Throughput (%) | 67.14 % | 58.22 % | -8.92 pp | 63.51 % | -3.63 pp | 58.05 % | -9.09 pp |
| Achieved Occupancy (%) | 94.18 % | 93.96 % | -0.22 pp | 94.45 % | +0.27 pp | 94.47 % | +0.29 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.03 % | 77.52 % | +6.49 pp | 71.43 % | +0.40 pp | 77.88 % | +6.85 pp |
| IPC Active | 1.17 inst/cycle | 0.90 inst/cycle | -0.27 | 1.15 inst/cycle | -0.02 | 0.92 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan A: `vectorized_stores_with_contiguous_hints_and_evict_first_loads` (0.809×)

Changing the eviction policy from evict_last to evict_first on a large streaming input tensor, combined with adding tl.max_contiguous/tl.multiple_of hints on the store address, caused a 24% latency regression (174us → 216us). Despite L2 throughput dropping from 97% to 94%, warp cycles per issued instruction worsened from 51.4 to 65.2, eligible warps per scheduler dropped from 0.57 to 0.43, and compute throughput fell from 29.7% to 23.2%. The evict_first policy appears to have reduced effective L2 reuse (L2 hit rate fell from 69.3% to 65.7%), increasing DRAM pressure and extending memory latency stalls rather than alleviating them. When the kernel is already heavily memory-latency-bound with high occupancy (93.85%), switching eviction policies on the primary load path can worsen stall depth without providing compensating bandwidth gains.

> **Applicable when**: Never — this combination caused regression in all tested conditions

> **Avoid when**: Kernel is memory-latency-bound with high occupancy (>90%), L2 hit rate ~69%, and the primary load tensor is accessed with a strided (non-contiguous) pattern. Switching eviction policy to evict_first on the strided load path worsens L2 hit rate and increases warp stall depth (Warp Cycles Per Issued Instruction rose from 51.4 to 65.2). Adding tl.max_contiguous/tl.multiple_of hints to the store address alone does not compensate. Do not apply when the load address is stride-96 (non-contiguous) — the contiguity hints are only valid for stride-1 accesses.

#### ❌ Plan C: `vectorized_stores_with_contiguous_hints_and_evict_first_loads` (0.809×)

Changing the eviction policy from evict_last to evict_first on a large streaming input tensor, combined with adding tl.max_contiguous/tl.multiple_of hints on the store address, caused a 24% latency regression (174us → 216us). Despite L2 throughput dropping from 97% to 94%, warp cycles per issued instruction worsened from 51.4 to 65.2, eligible warps per scheduler dropped from 0.57 to 0.43, and compute throughput fell from 29.7% to 23.2%. The evict_first policy appears to have reduced effective L2 reuse (L2 hit rate fell from 69.3% to 65.7%), increasing DRAM pressure and extending memory latency stalls rather than alleviating them. When the kernel is already heavily memory-latency-bound with high occupancy (93.85%), switching eviction policies on the primary load path can worsen stall depth without providing compensating bandwidth gains.

> **Applicable when**: Never — this combination caused regression in all tested conditions

> **Avoid when**: Kernel is memory-latency-bound with high occupancy (>90%), L2 hit rate ~69%, and the primary load tensor is accessed with a strided (non-contiguous) pattern. Switching eviction policy to evict_first on the strided load path worsens L2 hit rate and increases warp stall depth (Warp Cycles Per Issued Instruction rose from 51.4 to 65.2). Adding tl.max_contiguous/tl.multiple_of hints to the store address alone does not compensate. Do not apply when the load address is stride-96 (non-contiguous) — the contiguity hints are only valid for stride-1 accesses.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints_on_channel_dimension | 0.779× | 0.798× | ✓ |  |
| B | software_pipelining_to_hide_l1tex_latency_stalls | 0.999× | 1.000× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints_on_channel_axis | 0.997× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 174940 ns | 219230 ns | +44290 ns (+25.3%) | 174910 ns | -30 ns (-0.0%) | 176190 ns | +1250 ns (+0.7%) |
| SM Throughput (%) | 27.85 % | 22.63 % | -5.22 pp | 27.94 % | +0.09 pp | 29.73 % | +1.88 pp |
| Memory Throughput (%) | 81.19 % | 67.37 % | -13.82 pp | 81.51 % | +0.32 pp | 81.16 % | -0.03 pp |
| DRAM Throughput (%) | 67.14 % | 57.71 % | -9.43 pp | 67.15 % | +0.01 pp | 66.68 % | -0.46 pp |
| Achieved Occupancy (%) | 94.18 % | 94.10 % | -0.08 pp | 94.19 % | +0.01 pp | 93.79 % | -0.39 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 34.91 | 34.91 | +0.00 | 34.91 | +0.00 | 34.91 | +0.00 |
| No Eligible (%) | 71.03 % | 77.81 % | +6.78 pp | 71.00 % | -0.03 pp | 69.85 % | -1.18 pp |
| IPC Active | 1.17 inst/cycle | 0.95 inst/cycle | -0.22 | 1.16 inst/cycle | -0.01 | 1.16 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_with_contiguous_hints_on_channel_dimension` (0.798×)

Applying tl.max_contiguous + tl.multiple_of to the channel-dimension arange (c_off) and the store-side spatial arange (hw_off_store), combined with removing the eviction policy on stores, caused a 20% latency regression (184us → 219us). All memory metrics degraded: memory throughput dropped from 3.12 to 2.84 TB/s, L2 hit rate fell from 69.71% to 65.22%, and warp cycles per issued instruction worsened from 52.8 to 65.4. The contiguity hints on c_off did not produce vectorized loads because the subsequent modulo operation (y0 = c_off % 96) prevents the compiler from propagating the stride-1 hint through to the actual load address, leaving the uncoalesced access pattern unchanged while adding overhead. The additional store-side hint and eviction policy removal compounded the regression.

> **Applicable when**: Never — this combination caused regression in all measured metrics

> **Avoid when**: When the load address is derived from a contiguity-hinted index via a modulo operation (e.g., y0 = c_off % 96): the compiler cannot propagate the stride-1 hint through the modulo, so no vectorized loads are emitted and the hint provides no benefit. When the kernel is already memory-latency-bound with L2 hit rate ~69% and high occupancy (>94%), adding contiguous hints to indices that feed into non-linear address computations (modulo, floor-divide) will not improve coalescing and may increase stall depth. Do not apply tl.max_contiguous + tl.multiple_of to an index that is subsequently decomposed via modulo before use in a load address.

---

## Cumulative Progress

```
Baseline:  174940 ns  (1.000×)

Final best: 175420 ns  (0.9973×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,26 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 24576
     xnumel = 3025
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 96)
-    y1 = yindex // 96
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask)
-
-
+    # XBLOCK = HWBLOCK (fast-varying for stores, spatial dim, stride-1 in out_ptr0 NCHW)
+    # YBLOCK = CBLOCK  (fast-varying for loads, channel dim, stride-1 in in_ptr0 NHWC)
+    pid_hw = tl.program_id(0)
+    pid_c = tl.program_id(1)
+    # c_off: channel/batch index (ynumel=24576 = 256*96)
+    c_off = pid_c * YBLOCK + tl.arange(0, YBLOCK)
+    # hw_off: spatial index (xnumel=3025 = 55*55)
+    hw_off = pid_hw * XBLOCK + tl.arange(0, XBLOCK)
+    hw_mask = hw_off < xnumel
+    y0 = c_off % 96   # channel index within batch (0..95)
+    y1 = c_off // 96  # batch index (0..255)
+    # load_addr shape: [XBLOCK, YBLOCK] = [HWBLOCK, CBLOCK]
+    # in_ptr0 is NHWC: addr = y0 + 96*hw_off + 290400*y1
+    # YBLOCK (channel) is fast-varying -> coalesced loads
+    load_addr = y0[None, :] + 96 * hw_off[:, None] + 290400 * y1[None, :]
+    load_mask = hw_mask[:, None]
+    tile = tl.load(in_ptr0 + load_addr, load_mask, other=0.0, eviction_policy='evict_first')
+    bias = tl.load(in_ptr1 + y0, None, eviction_policy='evict_last')
+    tile = tile + bias[None, :]
+    # store_addr shape: [XBLOCK, YBLOCK] = [HWBLOCK, CBLOCK]
+    # out_ptr0 is NCHW: addr = hw_off + 3025*c_off
+    # XBLOCK (spatial) is fast-varying -> coalesced stores
+    store_addr = hw_off[:, None] + 3025 * c_off[None, :]
+    tl.store(out_ptr0 + store_addr, tile, hw_mask[:, None], eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
