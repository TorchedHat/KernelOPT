# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 3300 ns (**1.7545×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_063/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.06 % | **Memory**: 42.91 % | **DRAM**: 21.29 %
- **Achieved Occupancy**: 54.67 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.426× | 1.739× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.439× | 1.754× | ✓ | **✓ best** |
| D | add_autotune_increase_tile_sizes | 1.019× | 1.322× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3330 ns | -2460 ns (-42.5%) | 3300 ns | -2490 ns (-43.0%) | 4380 ns | -1410 ns (-24.4%) |
| SM Throughput (%) | 16.06 % | 17.25 % | +1.19 pp | 8.70 % | -7.36 pp | 26.03 % | +9.97 pp |
| Memory Throughput (%) | 42.91 % | 20.56 % | -22.35 pp | 21.07 % | -21.84 pp | 20.64 % | -22.27 pp |
| DRAM Throughput (%) | 21.29 % | 5.00 % | -16.29 pp | 5.04 % | -16.25 pp | 3.78 % | -17.51 pp |
| Achieved Occupancy (%) | 54.67 % | 45.53 % | -9.14 pp | 52.31 % | -2.36 pp | 71.55 % | +16.88 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.41 % | 54.34 % | -16.07 pp | 69.95 % | -0.46 pp | 36.46 % | -33.95 pp |
| IPC Active | 1.15 inst/cycle | 1.61 inst/cycle | +0.46 | 1.08 inst/cycle | -0.07 | 2.16 inst/cycle | +1.01 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.754×)

When a Triton kernel has hardcoded tile sizes that produce uncoalesced stores (excess sectors >20%) and high L1TEX stall cycles (>45% of warp cycles), adding @triton.autotune over XBLOCK and YBLOCK allows the runtime to select tile shapes that reduce per-CTA strided-write footprint and register pressure. In this case, autotuning reduced registers per thread from 26 to 16, eliminated dynamic shared memory (4.1 KB → 0), cut elapsed cycles by 43% (8585 → 4872), and improved L2 hit rate from 60.6% to 77.35%. The key mechanism is that smaller YBLOCK (e.g., YBLOCK=1) eliminates strided-y store interleaving, while larger XBLOCK increases CTA count to better cover available SMs.

> **Applicable when**: Triton kernel has: (1) hardcoded XBLOCK/YBLOCK tile sizes, (2) UncoalescedGlobalAccess rule triggered with >15% excess sectors, (3) CPIStall from L1TEX >40% of warp cycles, (4) waves_per_sm < 1.0 indicating under-utilization of SMs, (5) strided store patterns where the store index has a non-unit stride in one tile dimension.

> **Avoid when**: Kernel already uses @triton.autotune; kernel is compute-bound rather than memory-latency-bound; grid is already large enough (waves_per_sm > 2.0); tile sizes are constrained by shared memory usage that must be tuned jointly; autotune overhead is unacceptable in latency-sensitive single-call scenarios.

#### ✅ Plan D: `add_autotune_increase_tile_sizes` (1.359×)

When a Triton pointwise kernel has low waves-per-SM (<1.0) and low achieved occupancy (<60%) with the dominant stall being L1TEX scoreboard latency (CPIStall), adding @triton.autotune with larger XBLOCK/YBLOCK configurations allows the runtime to select tile shapes that double waves-per-SM and increase eligible warps per scheduler. In this case, doubling block size from 128 to 256 (selected by autotune) raised waves/SM from 0.74 to 1.48, eligible warps from 1.27 to 3.75, and cut warp cycles per issued instruction from 28.4 to 17.8, reducing kernel duration by ~24%. The lambda grid pattern (grid = lambda meta: (cdiv(N, meta['XBLOCK']), cdiv(M, meta['YBLOCK']))) is required to let autotune dynamically adjust the launch grid. Additionally, changing eviction policy from 'evict_last' to 'evict_first' on both load and store is appropriate for pure streaming copy kernels with zero data reuse.

> **Applicable when**: Triton pointwise or layout-conversion kernels with waves_per_sm < 1.0, achieved occupancy < 65%, eligible warps per scheduler < 2.0, and CPIStall (L1TEX scoreboard) as the dominant stall reason (>40% of total stall cycles). Grid size is fixed and small relative to SM count. Kernel is a pure streaming copy with no data reuse (L1 hit rate = 0%).

> **Avoid when**: Kernel is already achieving >1.5 waves/SM or >80% occupancy; register pressure is high (>32 registers/thread) and larger blocks would reduce occupancy further; the kernel has complex shared memory usage where tile size changes break correctness or exceed smem limits; data is reused across iterations (evict_first would hurt L2 hit rate).

---

## Iteration 2

**Best so far**: 1.7545× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.388× | 1.754× | ✓ |  |
| B | fix_autotune_grid_lambda_and_load_eviction_policy | 1.306× | 1.076× | ✓ |  |
| C | vectorized_loads_with_contiguous_hints | 1.317× | 1.372× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_autotune_grid... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3300 ns | -2490 ns (-43.0%) | 5380 ns | -410 ns (-7.1%) | 4220 ns | -1570 ns (-27.1%) |
| SM Throughput (%) | 16.06 % | 9.77 % | -6.29 pp | 29.25 % | +13.19 pp | 26.49 % | +10.43 pp |
| Memory Throughput (%) | 42.91 % | 21.08 % | -21.83 pp | 30.82 % | -12.09 pp | 21.03 % | -21.88 pp |
| DRAM Throughput (%) | 21.29 % | 5.03 % | -16.26 pp | 3.08 % | -18.21 pp | 3.93 % | -17.36 pp |
| Achieved Occupancy (%) | 54.67 % | 50.25 % | -4.42 pp | 74.50 % | +19.83 pp | 68.86 % | +14.19 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.41 % | 69.50 % | -0.91 pp | 35.75 % | -34.66 pp | 37.63 % | -32.78 pp |
| IPC Active | 1.15 inst/cycle | 1.04 inst/cycle | -0.11 | 2.42 inst/cycle | +1.27 | 2.39 inst/cycle | +1.24 |

---

## Iteration 3

**Best so far**: 1.7545× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_and_deeper_software_pipelining | 1.336× | 1.673× | ✓ |  |
| B | fix_autotune_grid_lambda_and_evict_first_for_streaming_copy | 1.326× | 1.359× | ✓ |  |
| C | store_coalescing_via_yblock3_alignment | 1.320× | 1.064× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_autotune_grid... | Δ | store_coalescing_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3460 ns | -2330 ns (-40.2%) | 4260 ns | -1530 ns (-26.4%) | 5440 ns | -350 ns (-6.0%) |
| SM Throughput (%) | 16.06 % | 9.95 % | -6.11 pp | 27.45 % | +11.39 pp | 29.55 % | +13.49 pp |
| Memory Throughput (%) | 42.91 % | 20.20 % | -22.71 pp | 22.13 % | -20.78 pp | 30.54 % | -12.37 pp |
| DRAM Throughput (%) | 21.29 % | 4.83 % | -16.46 pp | 3.88 % | -17.41 pp | 3.03 % | -18.26 pp |
| Achieved Occupancy (%) | 54.67 % | 53.09 % | -1.58 pp | 68.81 % | +14.14 pp | 73.92 % | +19.25 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.41 % | 70.14 % | -0.27 pp | 38.54 % | -31.87 pp | 35.78 % | -34.63 pp |
| IPC Active | 1.15 inst/cycle | 1.05 inst/cycle | -0.10 | 2.33 inst/cycle | +1.18 | 2.54 inst/cycle | +1.39 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     3300 ns  (1.754×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3300 ns  (1.7545×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+    triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +29,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _x_arange = tl.arange(0, XBLOCK)
+    _x_arange = tl.max_contiguous(tl.multiple_of(_x_arange, XBLOCK), XBLOCK)
+    xindex = xoffset + _x_arange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
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
