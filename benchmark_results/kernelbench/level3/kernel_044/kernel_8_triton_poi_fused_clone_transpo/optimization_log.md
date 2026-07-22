# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_transpose_view_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244060 ns — **Best latency**: 244220 ns (**0.9993×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_transpose_view_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_8_triton_poi_fused_clone_transpo/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244060 ns
- **SM Throughput**: 25.38 % | **Memory**: 32.07 % | **DRAM**: 32.07 %
- **Achieved Occupancy**: 43.41 %
- **IPC Active**: 1.02 inst/cycle | **No Eligible**: 74.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_transpose | 1.000× | 0.999× | ✓ | **✓ best** |
| B | vectorized_loads_with_autotune_xblock | 0.995× | 0.259× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_latency_hiding | 0.835× | 0.842× | ✓ |  |
| D | increase_xblock_via_autotune_to_hide_l1tex_latency | 0.801× | 0.807× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ | increase_xblock_v... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244060 ns | 244220 ns | +160 ns (+0.1%) | 943390 ns | +699330 ns (+286.5%) | 289760 ns | +45700 ns (+18.7%) | 302560 ns | +58500 ns (+24.0%) |
| SM Throughput (%) | 25.38 % | 29.29 % | +3.91 pp | 84.16 % | +58.78 pp | 46.58 % | +21.20 pp | 39.14 % | +13.76 pp |
| Memory Throughput (%) | 32.07 % | 31.81 % | -0.26 pp | 21.07 % | -11.00 pp | 26.88 % | -5.19 pp | 25.48 % | -6.59 pp |
| DRAM Throughput (%) | 32.07 % | 31.81 % | -0.26 pp | 8.18 % | -23.89 pp | 26.88 % | -5.19 pp | 25.48 % | -6.59 pp |
| Achieved Occupancy (%) | 43.41 % | 43.58 % | +0.17 pp | 78.04 % | +34.63 pp | 38.26 % | -5.15 pp | 33.40 % | -10.01 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 31.00 register/thread | +15 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 372.36 | +186.18 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 74.52 % | 70.21 % | -4.31 pp | 22.22 % | -52.30 pp | 50.85 % | -23.67 pp | 61.14 % | -13.38 pp |
| IPC Active | 1.02 inst/cycle | 1.19 inst/cycle | +0.17 | 3.11 inst/cycle | +2.09 | 1.90 inst/cycle | +0.88 | 1.53 inst/cycle | +0.51 |


### Key Learnings

#### ❌ Plan C: `autotune_xblock_and_num_warps_for_latency_hiding` (0.842×)

When a kernel is bottlenecked by non-coalesced strided memory accesses (0% L1 hit rate, high L1TEX scoreboard stall cycles), doubling the block size via autotune does not improve throughput and can regress performance. Although eligible warps per scheduler improved (0.30→0.74) and warp cycles per instruction dropped (26.6→12.2), the total instruction count doubled (48.8M→105.4M) and DRAM throughput fell (32%→27%), causing duration to increase from 244µs to 290µs. The root cause is that larger blocks issue more redundant address-computation instructions for the same data volume, and the strided access pattern prevents coalescing regardless of block size — the memory bottleneck is access pattern, not occupancy.

> **Applicable when**: Never applicable as a fix for strided/non-coalesced gather kernels where L1 hit rate is 0% and the bottleneck is access pattern rather than occupancy.

> **Avoid when**: Kernel has 0% L1 hit rate due to strided gather/transpose access pattern; increasing block size cannot fix non-coalescing and will increase total instruction count without improving memory throughput. Avoid when DRAM throughput is already below 35% and the stall is L1TEX scoreboard from non-coalesced loads — the fix must be the access pattern (e.g., shared memory tiling for transpose), not block size.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eviction_policies_streaming_gather_evict_first | 0.827× | 0.838× | ✓ |  |
| C | eviction_policies_streaming_load_store | 0.796× | 0.806× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 244060 ns | 291330 ns | +47270 ns (+19.4%) | 302880 ns | +58820 ns (+24.1%) |
| SM Throughput (%) | 25.38 % | 54.51 % | +29.13 pp | 43.93 % | +18.55 pp |
| Memory Throughput (%) | 32.07 % | 26.57 % | -5.50 pp | 24.60 % | -7.47 pp |
| DRAM Throughput (%) | 32.07 % | 26.57 % | -5.50 pp | 24.60 % | -7.47 pp |
| Achieved Occupancy (%) | 43.41 % | 38.80 % | -4.61 pp | 34.20 % | -9.21 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 186.18 | 372.36 | +186.18 | 186.18 | +0.00 |
| No Eligible (%) | 74.52 % | 44.74 % | -29.78 pp | 53.93 % | -20.59 pp |
| IPC Active | 1.02 inst/cycle | 2.16 inst/cycle | +1.14 | 1.81 inst/cycle | +0.79 |


### Key Learnings

#### ❌ Plan C: `eviction_policies_streaming_load_store` (0.806×)

Adding eviction_policy='evict_first' to loads and eviction_policy='evict_last' to stores in a strided transpose kernel caused an 0.81x regression. Despite reducing warp cycles per issued instruction (22.9→11.8) and improving scheduler eligibility, the executed instruction count nearly doubled (56.6M→104.2M), registers per thread increased from 16 to 18, achieved occupancy dropped from 43.6% to 34.2%, and DRAM throughput fell from 31.8% to 24.6%. The regression was compounded by the simultaneous addition of an @triton.autotune decorator with a fixed grid (not a lambda), which caused mismatched tile/grid sizing and inflated instruction count — the eviction policy change alone cannot be isolated as the sole cause, but the combined diff produced a clear regression.

> **Applicable when**: Never in this combined form. Eviction policy hints alone may be neutral, but combining them with autotune and a broken fixed grid causes regression.

> **Avoid when**: Do NOT combine eviction_policy hints with @triton.autotune when the grid is not updated to a lambda using meta['XBLOCK']. The fixed grid causes mismatched tile/grid sizing, nearly doubling instruction count. Additionally, do not apply eviction_policy='evict_last' on stores when the store is sequential (already cache-friendly) — it provides no benefit and the register pressure increase from autotune overhead reduces occupancy. For this specific strided transpose kernel (0% L1 hit rate, ~50% L2 hit rate, 186 waves/SM), eviction policy hints do not address the root cause (non-coalesced access pattern) and the autotune overhead compounds the regression.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_to_enable_xblock_variation | 0.827× | 0.838× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 244060 ns | 291170 ns | +47110 ns (+19.3%) |
| SM Throughput (%) | 25.38 % | 53.60 % | +28.22 pp |
| Memory Throughput (%) | 32.07 % | 26.60 % | -5.47 pp |
| DRAM Throughput (%) | 32.07 % | 26.60 % | -5.47 pp |
| Achieved Occupancy (%) | 43.41 % | 38.68 % | -4.73 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 372.36 | +186.18 |
| No Eligible (%) | 74.52 % | 43.98 % | -30.54 pp |
| IPC Active | 1.02 inst/cycle | 2.19 inst/cycle | +1.17 |

---

## Cumulative Progress

```
Baseline:  244060 ns  (1.000×)

Final best: 244220 ns  (0.9993×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 96)
     x1 = ((xindex // 96) % 8)
     x2 = ((xindex // 768) % 512)
     x3 = xindex // 393216
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), None)
-    tl.store(out_ptr0 + (x4), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), xmask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x4), tmp0, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648, XBLOCK=128)
+    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
