# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42590 ns — **Best latency**: 42660 ns (**0.9984×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_075/kernel_4_triton_poi_fused_add_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 16
- **Duration**: 42590 ns
- **SM Throughput**: 24.86 % | **Memory**: 16.48 % | **DRAM**: 3.48 %
- **Achieved Occupancy**: 31.95 %
- **IPC Active**: 1.06 inst/cycle | **No Eligible**: 73.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_broadcast_add | 0.995× | 0.995× | ✓ |  |
| B | increase_xblock_with_autotune_for_broadcast_add | 1.001× | 0.995× | ✓ | **✓ best** |
| D | vectorized_loads_larger_xblock | 0.997× | 0.685× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42590 ns | 42820 ns | +230 ns (+0.5%) | 42820 ns | +230 ns (+0.5%) | 62140 ns | +19550 ns (+45.9%) |
| SM Throughput (%) | 24.86 % | 41.61 % | +16.75 pp | 40.68 % | +15.82 pp | 87.80 % | +62.94 pp |
| Memory Throughput (%) | 16.48 % | 25.80 % | +9.32 pp | 25.67 % | +9.19 pp | 56.08 % | +39.60 pp |
| DRAM Throughput (%) | 3.48 % | 3.40 % | -0.08 pp | 3.29 % | -0.19 pp | 2.25 % | -1.23 pp |
| Achieved Occupancy (%) | 31.95 % | 20.55 % | -11.40 pp | 21.20 % | -10.75 pp | 78.82 % | +46.87 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 24.00 register/thread | +8 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 62.06 | +31.03 |
| No Eligible (%) | 73.46 % | 51.84 % | -21.62 pp | 53.74 % | -19.72 pp | 9.32 % | -64.14 pp |
| IPC Active | 1.06 inst/cycle | 1.88 inst/cycle | +0.82 | 1.83 inst/cycle | +0.77 | 3.62 inst/cycle | +2.56 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.023× | 0.022× | ✓ |  |
| B | persistent_tile_scheduling | 0.186× | 0.174× | ✓ |  |
| C | persistent_tile_scheduling_with_fixed_autotune_grid | 1.004× | 0.998× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.001× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42590 ns | 1960000 ns | +1917410 ns (+4502.0%) | 244610 ns | +202020 ns (+474.3%) | 42660 ns | +70 ns (+0.2%) | 42620 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 24.86 % | 91.21 % | +66.35 pp | 17.31 % | -7.55 pp | 34.13 % | +9.27 pp | 27.94 % | +3.08 pp |
| Memory Throughput (%) | 16.48 % | 83.55 % | +67.07 pp | 79.99 % | +63.51 pp | 16.47 % | -0.01 pp | 16.48 % | +0.00 pp |
| DRAM Throughput (%) | 3.48 % | 3.75 % | +0.27 pp | 1.50 % | -1.98 pp | 3.36 % | -0.12 pp | 3.37 % | -0.11 pp |
| Achieved Occupancy (%) | 31.95 % | 99.42 % | +67.47 pp | 90.41 % | +58.46 pp | 36.06 % | +4.11 pp | 33.52 % | +1.57 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 28.00 register/thread | +12 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 73.46 % | 24.39 % | -49.07 pp | 89.38 % | +15.92 pp | 63.63 % | -9.83 pp | 70.27 % | -3.19 pp |
| IPC Active | 1.06 inst/cycle | 3.02 inst/cycle | +1.96 | 0.42 inst/cycle | -0.64 | 1.45 inst/cycle | +0.39 | 1.18 inst/cycle | +0.12 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.174×)

Replacing a flat grid with a persistent kernel (while-loop over tiles) caused a 5.7x slowdown (42.8us → 244.6us) because the grid was never updated from (cdiv(xnumel, 128),) = 65536 CTAs to the intended 132 CTAs. The persistent kernel body ran inside 65536 CTAs each iterating the while loop, causing L2 traffic to explode (L2 Compression Input Sectors: 1M → 30.7M, L2 Cache Throughput: 17% → 84%) and Warp Cycles Per Issued Instruction to skyrocket from 6.6 to 138.8 cycles due to LG queue full stalls (50.6% of warp cycles). Despite occupancy improving from 20.6% to 90.4%, the massive increase in memory traffic and instruction count (14M → 18.8M) made the kernel far slower — the persistent loop ran ~62 iterations per CTA across all 65536 CTAs instead of only 132.

> **Applicable when**: Never — this is a pure implementation flaw. The persistent kernel concept is sound but the grid was not updated.

> **Avoid when**: Always verify the grid launch site is updated to match the persistent kernel's intended CTA count. When converting a flat kernel to a persistent kernel, the grid MUST change from (cdiv(N, BLOCK),) to (NUM_SMS,). Failing to update the grid causes all CTAs with program_id >= total_tiles to load in_ptr0 (TILE_SIZE floats) wastefully and then immediately exit the while loop, while CTAs with program_id < total_tiles each process only a single tile instead of ~62 — the opposite of the intended behavior. The register pressure increase (16 → 28 regs/thread) from holding TILE_SIZE=1024 elements in registers also reduces Block Limit Registers from 32 to 16, which can cap occupancy in other contexts.

#### ❌ Plan B: `persistent_tile_scheduling` (0.174×)

Replacing a flat grid with a persistent kernel (while-loop over tiles) caused a 5.7x slowdown (42.8us → 244.6us) because the grid was never updated from (cdiv(xnumel, 128),) = 65536 CTAs to the intended 132 CTAs. The persistent kernel body ran inside 65536 CTAs each iterating the while loop, causing L2 traffic to explode (L2 Compression Input Sectors: 1M → 30.7M, L2 Cache Throughput: 17% → 84%) and Warp Cycles Per Issued Instruction to skyrocket from 6.6 to 138.8 cycles due to LG queue full stalls (50.6% of warp cycles). Despite occupancy improving from 20.6% to 90.4%, the massive increase in memory traffic and instruction count (14M → 18.8M) made the kernel far slower — the persistent loop ran ~62 iterations per CTA across all 65536 CTAs instead of only 132.

> **Applicable when**: Never — this is a pure implementation flaw. The persistent kernel concept is sound but the grid was not updated.

> **Avoid when**: Always verify the grid launch site is updated to match the persistent kernel's intended CTA count. When converting a flat kernel to a persistent kernel, the grid MUST change from (cdiv(N, BLOCK),) to (NUM_SMS,). Failing to update the grid causes all CTAs with program_id >= total_tiles to load in_ptr0 (TILE_SIZE floats) wastefully and then immediately exit the while loop, while CTAs with program_id < total_tiles each process only a single tile instead of ~62 — the opposite of the intended behavior. The register pressure increase (16 → 28 regs/thread) from holding TILE_SIZE=1024 elements in registers also reduces Block Limit Registers from 32 to 16, which can cap occupancy in other contexts.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | revert_broken_persistent_kernel_to_flat_elementwise | 1.001× | 0.996× | ✓ |  |
| D | revert_broken_persistent_kernel_to_flat_elementwise | 1.002× | 1.001× | ✓ |  |
| E | add_triton_autotune | 0.181× | 0.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_broken_per... | Δ | revert_broken_per... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 42590 ns | 42750 ns | +160 ns (+0.4%) | 42560 ns | -30 ns (-0.1%) | 1960000 ns | +1917410 ns (+4502.0%) |
| SM Throughput (%) | 24.86 % | 20.12 % | -4.74 pp | 20.07 % | -4.79 pp | 91.21 % | +66.35 pp |
| Memory Throughput (%) | 16.48 % | 16.42 % | -0.06 pp | 16.44 % | -0.04 pp | 83.54 % | +67.06 pp |
| DRAM Throughput (%) | 3.48 % | 3.44 % | -0.04 pp | 3.44 % | -0.04 pp | 3.76 % | +0.28 pp |
| Achieved Occupancy (%) | 31.95 % | 32.17 % | +0.22 pp | 32.48 % | +0.53 pp | 99.43 % | +67.48 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 73.46 % | 78.30 % | +4.84 pp | 78.46 % | +5.00 pp | 24.44 % | -49.02 pp |
| IPC Active | 1.06 inst/cycle | 0.86 inst/cycle | -0.20 | 0.86 inst/cycle | -0.20 | 3.02 inst/cycle | +1.96 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cache_eviction_hints_for_broadcast_operand_reuse | 0.181× | 0.022× | ✓ |  |
| D | add_triton_autotune | 0.181× | 0.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cache_eviction_hi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42590 ns | 1960000 ns | +1917410 ns (+4502.0%) | 1960000 ns | +1917410 ns (+4502.0%) |
| SM Throughput (%) | 24.86 % | 91.03 % | +66.17 pp | 91.20 % | +66.34 pp |
| Memory Throughput (%) | 16.48 % | 83.39 % | +66.91 pp | 83.59 % | +67.11 pp |
| DRAM Throughput (%) | 3.48 % | 4.75 % | +1.27 pp | 3.70 % | +0.22 pp |
| Achieved Occupancy (%) | 31.95 % | 99.43 % | +67.48 pp | 99.43 % | +67.48 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 73.46 % | 24.17 % | -49.29 pp | 24.42 % | -49.04 pp |
| IPC Active | 1.06 inst/cycle | 3.03 inst/cycle | +1.97 | 3.02 inst/cycle | +1.96 |

---

## Cumulative Progress

```
Baseline:  42590 ns  (1.000×)

Final best: 42660 ns  (0.9984×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,21 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
-def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 8388608
+def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
-    x0 = (xindex % 1024)
-    x1 = xindex // 1024
-    x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2), tmp2, None)
-
-
+    xindex = xoffset + tl.arange(0, XBLOCK)  # full tile, no mask needed
+    x0 = xindex % 1024   # column index into in_ptr0 (shape: 1024)
+    x1 = xindex // 1024  # row index into in_ptr1 (shape: 8192)
+    tmp0 = tl.load(in_ptr0 + x0)
+    tmp1 = tl.load(in_ptr1 + x1)
+    tl.store(out_ptr0 + xindex, tmp0 + tmp1)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
