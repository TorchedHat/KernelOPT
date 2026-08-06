# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6210 ns — **Best latency**: 6140 ns (**1.0114×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_to_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6210 ns
- **SM Throughput**: 16.16 % | **Memory**: 39.85 % | **DRAM**: 19.92 %
- **Achieved Occupancy**: 54.79 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.127× | 1.356× | ✓ |  |
| B | autotune_tile_sizes_and_warps_to_hide_l1tex_latency | 1.384× | 1.011× | ✓ | **✓ best** |
| D | add_autotune_increase_tile_sizes | 1.279× | 1.515× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 4580 ns | -1630 ns (-26.2%) | 6140 ns | -70 ns (-1.1%) | 4100 ns | -2110 ns (-34.0%) |
| SM Throughput (%) | 16.16 % | 21.06 % | +4.90 pp | 16.29 % | +0.13 pp | 19.54 % | +3.38 pp |
| Memory Throughput (%) | 39.85 % | 30.65 % | -9.20 pp | 40.71 % | +0.86 pp | 22.65 % | -17.20 pp |
| DRAM Throughput (%) | 19.92 % | 14.48 % | -5.44 pp | 20.11 % | +0.19 pp | 4.04 % | -15.88 pp |
| Achieved Occupancy (%) | 54.79 % | 47.29 % | -7.50 pp | 55.47 % | +0.68 pp | 48.91 % | -5.88 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 26.00 register/thread | +0 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.83 % | 57.42 % | -13.41 pp | 71.12 % | +0.29 pp | 51.32 % | -19.51 pp |
| IPC Active | 1.15 inst/cycle | 1.68 inst/cycle | +0.53 | 1.12 inst/cycle | -0.03 | 1.83 inst/cycle | +0.68 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguous_hints` (1.882×)

Applying tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 4), 4) to the contiguous load dimension proves 128-bit alignment to the Triton compiler, enabling vectorized 4×FP32 loads and reducing register pressure (25→18 registers/thread). Combined with changing eviction_policy from 'evict_last' to 'evict_first' on streaming read-once loads, and refreshing autotune configs to include higher num_stages (2–3) for better software pipelining, total executed instructions dropped from 609K to 430K and kernel duration fell from 4.1 µs to 3.3 µs (1.88× speedup). The causal chain is: alignment hints → vectorized loads → fewer instructions → lower register pressure → faster execution, even when the store-side uncoalesced access pattern (stride-3 scatter) cannot be fixed.

> **Applicable when**: Triton kernel loads from a stride-1 (contiguous) dimension; the contiguous dimension size is divisible by 4 (for 128-bit / 4×FP32 alignment); data is read once per kernel (streaming pattern suits evict_first); kernel already has autotuning but with num_stages=1 configs only; UncoalescedGlobalAccess or CPIStall rules fire; register pressure is high (>20 registers/thread).

> **Avoid when**: Load address is not stride-1 contiguous (e.g., strided or indirect accesses); XBLOCK < 4 (hint value must not exceed block size); data is reused across iterations within the kernel (evict_first would hurt cache reuse); the dimension size is not divisible by the hint value N (would produce incorrect alignment claims).

---

## Iteration 2

**Best so far**: 1.0114× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_transpose_for_coalesced_nhwc_store | 1.275× | 1.472× | ✓ |  |
| B | software_pipelining_via_num_stages_autotune | 1.180× | 1.368× | ✓ |  |
| C | vectorized_loads_channel_aligned_tiles_expanded_autotune | 1.326× | 1.882× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ | software_pipelini... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 4220 ns | -1990 ns (-32.0%) | 4540 ns | -1670 ns (-26.9%) | 3300 ns | -2910 ns (-46.9%) |
| SM Throughput (%) | 16.16 % | 18.00 % | +1.84 pp | 19.93 % | +3.77 pp | 17.41 % | +1.25 pp |
| Memory Throughput (%) | 39.85 % | 20.78 % | -19.07 pp | 30.80 % | -9.05 pp | 20.77 % | -19.08 pp |
| DRAM Throughput (%) | 19.92 % | 3.92 % | -16.00 pp | 14.53 % | -5.39 pp | 5.01 % | -14.91 pp |
| Achieved Occupancy (%) | 54.79 % | 48.87 % | -5.92 pp | 46.78 % | -8.01 pp | 43.89 % | -10.90 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 27.00 register/thread | +1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.83 % | 50.55 % | -20.28 pp | 56.84 % | -13.99 pp | 54.38 % | -16.45 pp |
| IPC Active | 1.15 inst/cycle | 1.83 inst/cycle | +0.68 | 1.69 inst/cycle | +0.54 | 1.67 inst/cycle | +0.52 |


### Key Learnings

#### ✅ Plan B: `software_pipelining_via_num_stages_autotune` (1.368×)

When a Triton pointwise/transpose kernel already has @triton.autotune but only uses num_stages=1, expanding the config list to include num_stages=2 and num_stages=3 enables Triton's software pipelining (async cp.async prefetch), which is intended to overlap memory loads with compute and reduce scheduler stalls. However, in this case the profiling data shows the change actually increased elapsed cycles (6100→6745), worsened No Eligible (51.3%→56.8%), and reduced L2 hit rate (76.2%→62.8%), while L1 hit rate improved from 0% to 26.6% and memory throughput jumped from 197 to 708 GB/s — suggesting the autotuner selected a config that increased total memory traffic. The net wall-clock speedup of 1.37x was achieved despite worse scheduler metrics, likely because the selected config (larger XBLOCK or higher num_stages) changed the access pattern in a way that benefited the specific problem size, but the pipelining itself did not reduce stalls as predicted.

> **Applicable when**: Triton kernel already uses @triton.autotune with only num_stages=1 configs; IssueSlotUtilization rule is triggered with No Eligible > 50%; kernel is pointwise or layout-transpose with no shared memory tiling; L1 hit rate is near 0% suggesting no cache reuse benefit from current tile shape.

> **Avoid when**: Kernel already explores num_stages=2/3 in its autotune configs; kernel uses shared memory tiling where num_stages controls SMEM double-buffering and increasing it would exceed SMEM limits; the dominant stall is not memory latency but compute-bound (Issue Slots Busy > 80%); problem size is so small that autotuning overhead dominates runtime.

---

## Iteration 3

**Best so far**: 1.0114× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staging_for_coalesced_nhwc_store | 1.008× | 0.847× | ✓ |  |
| C | fix_autotune_grid_to_use_meta_dict_for_effective_tile_exploration | 1.230× | 1.377× | ✓ |  |
| D | add_triton_autotune | 1.327× | 1.515× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 7330 ns | +1120 ns (+18.0%) | 4510 ns | -1700 ns (-27.4%) | 4100 ns | -2110 ns (-34.0%) |
| SM Throughput (%) | 16.16 % | 43.23 % | +27.07 pp | 21.32 % | +5.16 pp | 16.84 % | +0.68 pp |
| Memory Throughput (%) | 39.85 % | 40.43 % | +0.58 pp | 31.23 % | -8.62 pp | 33.47 % | -6.38 pp |
| DRAM Throughput (%) | 19.92 % | 8.95 % | -10.97 pp | 14.65 % | -5.27 pp | 16.08 % | -3.84 pp |
| Achieved Occupancy (%) | 54.79 % | 76.05 % | +21.26 pp | 46.14 % | -8.65 pp | 53.31 % | -1.48 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 27.00 register/thread | +1 | 21.00 register/thread | -5 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.83 % | 36.14 % | -34.69 pp | 55.96 % | -14.87 pp | 64.85 % | -5.98 pp |
| IPC Active | 1.15 inst/cycle | 2.61 inst/cycle | +1.46 | 1.68 inst/cycle | +0.53 | 1.33 inst/cycle | +0.18 |


### Key Learnings

#### ❌ Plan A: `shared_memory_staging_for_coalesced_nhwc_store` (0.847×)

Replacing a stride-3 scatter store with a shared-memory staging pattern using tl.static_range(3) to emit per-channel coalesced stores caused a 2.2x increase in kernel duration (3.3 µs → 7.33 µs) and a 5.9x increase in executed instructions (430K → 2.54M). The static_range(3) loop tripled the number of store instructions issued (one full-tile store per channel iteration), and the larger block size (128→256 threads) combined with higher register pressure (18→32 registers/thread) constrained occupancy gains despite waves_per_sm doubling. The uncoalesced access warning remained at 50% excessive sectors, indicating the chan_mask=(y0==c) approach did not actually coalesce stores — threads in a warp still scatter across non-contiguous addresses because y0 varies across the YBLOCK dimension, not the XBLOCK (warp) dimension.

> **Applicable when**: Never: this pattern is strictly worse than the single uncoalesced store for this kernel.

> **Avoid when**: When the channel dimension (y0) varies across the YBLOCK (row) dimension rather than the XBLOCK (warp/column) dimension. The tl.static_range(c) + chan_mask approach does NOT coalesce stores when channel index is in the slow-varying tile dimension — it only masks out rows, leaving the store address stride-3 in x unchanged. This multiplies instruction count by the number of channels (3x here) with no coalescing benefit. Avoid when: (1) the uncoalesced dimension is x (warp-lane dimension) and the channel index is in y (YBLOCK rows); (2) the fix would multiply executed instructions by the channel count; (3) register pressure would increase significantly (>10 registers/thread increase).

---

## Cumulative Progress

```
Baseline:  6210 ns  (1.000×)
Iter 1     6140 ns  (1.011×)  — autotune_tile_sizes_and_warps_to_hide_l1tex_latency

Final best: 6140 ns  (1.0114×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +29,15 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 4), 4)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
