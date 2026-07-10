# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285630 ns — **Best latency**: 285790 ns (**0.9994×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_016/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285630 ns
- **SM Throughput**: 19.72 % | **Memory**: 78.87 % | **DRAM**: 72.25 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_xblock_yblock | 1.000× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285630 ns | 285790 ns | +160 ns (+0.1%) |
| SM Throughput (%) | 19.72 % | 19.83 % | +0.11 pp |
| Memory Throughput (%) | 78.87 % | 78.91 % | +0.04 pp |
| DRAM Throughput (%) | 72.25 % | 72.20 % | -0.05 pp |
| Achieved Occupancy (%) | 94.04 % | 94.08 % | +0.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.01 % | 80.33 % | +0.32 pp |
| IPC Active | 0.79 inst/cycle | 0.78 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_policy_fix_and_coalesced_load_via_tile_reshape | 0.691× | 0.695× | ✓ |  |
| B | eviction_policy_fix_and_yblock_alignment_to_channel_size | 0.691× | 0.700× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_policy_fix_... | Δ | eviction_policy_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285630 ns | 411200 ns | +125570 ns (+44.0%) | 408160 ns | +122530 ns (+42.9%) |
| SM Throughput (%) | 19.72 % | 13.28 % | -6.44 pp | 13.38 % | -6.34 pp |
| Memory Throughput (%) | 78.87 % | 59.99 % | -18.88 pp | 60.43 % | -18.44 pp |
| DRAM Throughput (%) | 72.25 % | 57.64 % | -14.61 pp | 58.06 % | -14.19 pp |
| Achieved Occupancy (%) | 94.04 % | 94.88 % | +0.84 pp | 94.94 % | +0.90 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.01 % | 86.51 % | +6.50 pp | 86.52 % | +6.51 pp |
| IPC Active | 0.79 inst/cycle | 0.54 inst/cycle | -0.25 | 0.54 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan A: `evict_policy_fix_and_coalesced_load_via_tile_reshape` (0.695×)

Switching the eviction policy from 'evict_last' to 'evict_first' on a strided, non-coalesced load in a memory-bound transpose kernel caused a 44% increase in kernel duration (285 µs → 411 µs). Warp cycles per issued instruction nearly doubled (75.5 → 112.4 cycles), eligible warps per scheduler dropped (0.28 → 0.20), and both L2 throughput and DRAM throughput fell, indicating the change worsened latency hiding rather than improving it. The tile reshape (XBLOCK=128→1, YBLOCK=8→128) was not applied in this iteration — only the eviction policy changed — so the regression is attributable solely to 'evict_first' increasing effective memory latency by preventing L1/L2 from retaining lines that were still being reused across warps.

> **Applicable when**: Never: this specific change (evict_last → evict_first on a strided non-coalesced load without accompanying tile reshape) is harmful.

> **Avoid when**: Do not change eviction policy from 'evict_last' to 'evict_first' on strided, non-coalesced loads in memory-bound kernels without simultaneously fixing the access pattern (e.g., tile reshape to achieve coalescing). When L2 hit rate is ~67% and warp stalls are dominated by L1TEX scoreboard waits, 'evict_first' removes the L2 residency that was providing that hit rate, increasing effective DRAM latency and worsening stall cycles per instruction from 75 to 112 cycles.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_store_hints_via_max_contiguous_multiple_of | 0.691× | 0.700× | ✓ |  |
| C | vectorized_store_hints_via_max_contiguous_multiple_of | 0.690× | 0.700× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_store_... | Δ | vectorized_store_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285630 ns | 408260 ns | +122630 ns (+42.9%) | 408160 ns | +122530 ns (+42.9%) |
| SM Throughput (%) | 19.72 % | 13.22 % | -6.50 pp | 13.47 % | -6.25 pp |
| Memory Throughput (%) | 78.87 % | 60.43 % | -18.44 pp | 60.44 % | -18.43 pp |
| DRAM Throughput (%) | 72.25 % | 58.03 % | -14.22 pp | 58.01 % | -14.24 pp |
| Achieved Occupancy (%) | 94.04 % | 94.89 % | +0.85 pp | 94.90 % | +0.86 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.01 % | 86.44 % | +6.43 pp | 86.49 % | +6.48 pp |
| IPC Active | 0.79 inst/cycle | 0.54 inst/cycle | -0.25 | 0.54 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan A: `vectorized_store_hints_via_max_contiguous_multiple_of` (0.700×)

Wrapping the store index with tl.max_contiguous and tl.multiple_of to hint vectorized 128-bit store transactions produced no measurable improvement: kernel duration remained identical at 408 µs, DRAM throughput barely changed (58.06% → 58.01%), and all memory metrics were statistically flat. The bottleneck is the non-coalesced strided LOAD path (128-float stride between threads), which dominates warp stall cycles (~110 cycles/instruction from L1TEX scoreboard waits); optimizing the already-coalesced store path has negligible impact when the load path is the binding constraint. Adding these hints to xindex also affected the load index computation (since xindex is shared), potentially introducing unintended side effects without any benefit.

> **Applicable when**: Never for this kernel pattern: when the primary bottleneck is a non-coalesced strided load (L1TEX stall cycles ~110/instruction, eligible warps ~0.2/scheduler), optimizing the store path with vectorization hints does not move any profiling metric.

> **Avoid when**: Do not apply tl.max_contiguous/tl.multiple_of store-side hints when the kernel is bottlenecked by non-coalesced loads rather than stores. When warp stall cycles per issued instruction exceed 100 and are attributed to L1TEX scoreboard waits on the load path, store-side vectorization hints will not reduce kernel duration. Additionally, applying these hints to the shared xindex variable affects both load and store index computations simultaneously, which may be unintended.

#### ❌ Plan C: `vectorized_store_hints_via_max_contiguous_multiple_of` (0.700×)

Wrapping the store index with tl.max_contiguous and tl.multiple_of to hint vectorized 128-bit store transactions produced no measurable improvement: kernel duration remained identical at 408 µs, DRAM throughput barely changed (58.06% → 58.01%), and all memory metrics were statistically flat. The bottleneck is the non-coalesced strided LOAD path (128-float stride between threads), which dominates warp stall cycles (~110 cycles/instruction from L1TEX scoreboard waits); optimizing the already-coalesced store path has negligible impact when the load path is the binding constraint. Adding these hints to xindex also affected the load index computation (since xindex is shared), potentially introducing unintended side effects without any benefit.

> **Applicable when**: Never for this kernel pattern: when the primary bottleneck is a non-coalesced strided load (L1TEX stall cycles ~110/instruction, eligible warps ~0.2/scheduler), optimizing the store path with vectorization hints does not move any profiling metric.

> **Avoid when**: Do not apply tl.max_contiguous/tl.multiple_of store-side hints when the kernel is bottlenecked by non-coalesced loads rather than stores. When warp stall cycles per issued instruction exceed 100 and are attributed to L1TEX scoreboard waits on the load path, store-side vectorization hints will not reduce kernel duration. Additionally, applying these hints to the shared xindex variable affects both load and store index computations simultaneously, which may be unintended.

---

## Cumulative Progress

```
Baseline:  285630 ns  (1.000×)

Final best: 285790 ns  (0.9994×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), mask=xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
