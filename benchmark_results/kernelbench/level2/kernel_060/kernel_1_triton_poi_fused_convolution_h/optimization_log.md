# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_hardswish_mul_native_group_norm_sigmoid_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1220000 ns — **Best latency**: 1210000 ns (**1.0083×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_tl_exp_with_libdevice_fast_expf_in_sigmoid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_hardswish_mul_native_group_norm_sigmoid_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_060/kernel_1_triton_poi_fused_convolution_h/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1968624,1,1), Waves/SM: 932.11
- **Registers/thread**: 22
- **Duration**: 1220000 ns
- **SM Throughput**: 73.77 % | **Memory**: 33.43 % | **DRAM**: 33.43 %
- **Achieved Occupancy**: 68.38 %
- **IPC Active**: 2.95 inst/cycle | **No Eligible**: 26.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_num_warps_to_improve_occupancy_and_reduce_l1tex_stalls | 1.000× | 0.629× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 1.000× | 1.000× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 1.000× | 0.169× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1220000 ns | 1940000 ns | +720000 ns (+59.0%) | 1220000 ns | +0 ns (+0.0%) | 7240000 ns | +6020000 ns (+493.4%) |
| SM Throughput (%) | 73.77 % | 78.09 % | +4.32 pp | 73.77 % | +0.00 pp | 77.44 % | +3.67 pp |
| Memory Throughput (%) | 33.43 % | 28.80 % | -4.63 pp | 33.44 % | +0.01 pp | 25.39 % | -8.04 pp |
| DRAM Throughput (%) | 33.43 % | 20.99 % | -12.44 pp | 33.44 % | +0.01 pp | 5.62 % | -27.81 pp |
| Achieved Occupancy (%) | 68.38 % | 84.58 % | +16.20 pp | 68.54 % | +0.16 pp | 90.30 % | +21.92 pp |
| Registers/Thread | 22.00 register/thread | 30.00 register/thread | +8 | 22.00 register/thread | +0 | 32.00 register/thread | +10 |
| Waves/SM | 932.11 | 932.11 | +0.00 | 932.11 | +0.00 | 1864.23 | +932.12 |
| No Eligible (%) | 26.21 % | 21.83 % | -4.38 pp | 26.07 % | -0.14 pp | 22.53 % | -3.68 pp |
| IPC Active | 2.95 inst/cycle | 3.13 inst/cycle | +0.18 | 2.95 inst/cycle | +0.00 | 3.10 inst/cycle | +0.15 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_num_warps_to_improve_occupancy_and_reduce_l1tex_stalls` (0.629×)

Adding @triton.autotune with larger XBLOCK configs (256–1024) increased achieved occupancy from 68.4% to 84.6% and reduced DRAM throughput, but caused a 59% slowdown. The selected config increased registers per thread from 22 to 30, which reduced block-level parallelism and caused executed instructions to jump from 709M to 1197M (69% more work). The dominant stall shifted from L1TEX scoreboard (latency) to 'not selected' warp scheduling, indicating the kernel became over-subscribed with warps rather than latency-limited, and the extra register pressure plus instruction overhead outweighed any occupancy benefit.

> **Applicable when**: Never: this specific pattern (autotuning XBLOCK upward for pointwise kernels with moderate register pressure ~22 regs/thread and high wave count ~932 waves/SM) caused regression here.

> **Avoid when**: Do NOT apply XBLOCK autotune upward when: (1) the kernel already has high wave count (>100 waves/SM), meaning grid is already large enough to saturate the GPU; (2) register pressure is near the boundary where larger XBLOCK configs will push registers/thread higher (22→30), reducing block occupancy; (3) the dominant stall is L1TEX scoreboard but the kernel is already compute-bound (>70% SM SOL) — increasing XBLOCK shifts the bottleneck to warp scheduling ('not selected') without reducing actual memory latency; (4) instruction count is sensitive to XBLOCK (e.g., boundary-check predicates scale with XBLOCK size).

#### ❌ Plan D: `autotune_xblock_increase_tile_size` (0.169×)

Adding @triton.autotune over larger XBLOCK values (256–1024) caused a 6x regression (1.22 ms → 7.24 ms) despite occupancy rising from 68% to 90%. The autotuner selected XBLOCK=256 (block_size doubled), but registers per thread jumped from 22 to 32, and executed instructions ballooned from 708M to 4.4B (6.3x increase), indicating the compiler generated significantly more code per tile — likely due to loop unrolling or additional predicate/mask logic at larger tile sizes. The stall pattern also shifted from L1TEX scoreboard stalls (44.6% before) to 'not selected' warp stalls (48.8% after), meaning the bottleneck moved but total work exploded. Do not apply autotune-based XBLOCK increases when the kernel already has very high waves/SM (>900) and the register count is near the block-limit boundary, as the compiler may generate substantially more instructions per thread at larger tile sizes.

> **Applicable when**: Never — this specific pattern caused severe regression. Autotune over XBLOCK may be valid in other kernels but not when: (1) the kernel already has high waves/SM (>900), (2) register count is near block-limit boundaries, and (3) the compiler generates substantially more instructions at larger tile sizes.

> **Avoid when**: Kernel has very high waves/SM (>900), is already compute-bound (SM SOL >70%), and has low register pressure (≤22 regs/thread) at the baseline tile size. In these cases, increasing XBLOCK via autotune causes the compiler to generate far more instructions per thread (observed 6x increase), overwhelming any latency-hiding benefit from higher occupancy. Also avoid when the grid is fixed and not updated to use meta['XBLOCK'] — the diff shows the grid was NOT updated to lambda meta, meaning the autotuner's selected XBLOCK was mismatched with the launch grid.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_libdevice_rsqrt_and_reduce_nonfused_fp32_instructions | 1.000× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1220000 ns | 1210000 ns | -10000 ns (-0.8%) |
| SM Throughput (%) | 73.77 % | 73.66 % | -0.11 pp |
| Memory Throughput (%) | 33.43 % | 33.46 % | +0.03 pp |
| DRAM Throughput (%) | 33.43 % | 33.46 % | +0.03 pp |
| Achieved Occupancy (%) | 68.38 % | 68.43 % | +0.05 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 |
| Waves/SM | 932.11 | 932.11 | +0.00 |
| No Eligible (%) | 26.21 % | 26.12 % | -0.09 pp |
| IPC Active | 2.95 inst/cycle | 2.95 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0083× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cache_eviction_policy_tuning_for_streaming_access_pattern | 1.000× | 1.000× | ✓ |  |
| B | streaming_store_evict_first_to_free_l2_for_broadcast_inputs | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_stores_with_eviction_policy_hints | 0.999× | 1.000× | ✓ |  |
| D | reduce_num_warps_to_improve_block_level_parallelism | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | cache_eviction_po... | Δ | streaming_store_e... | Δ | vectorized_loads_... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1220000 ns | 1220000 ns | +0 ns (+0.0%) | 1220000 ns | +0 ns (+0.0%) | 1220000 ns | +0 ns (+0.0%) | 1220000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 73.77 % | 77.70 % | +3.93 pp | 73.78 % | +0.01 pp | 77.87 % | +4.10 pp | 73.65 % | -0.12 pp |
| Memory Throughput (%) | 33.43 % | 33.40 % | -0.03 pp | 33.37 % | -0.06 pp | 33.32 % | -0.11 pp | 33.43 % | +0.00 pp |
| DRAM Throughput (%) | 33.43 % | 33.40 % | -0.03 pp | 33.37 % | -0.06 pp | 33.32 % | -0.11 pp | 33.43 % | +0.00 pp |
| Achieved Occupancy (%) | 68.38 % | 73.02 % | +4.64 pp | 68.41 % | +0.03 pp | 72.28 % | +3.90 pp | 68.68 % | +0.30 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 932.11 | 932.11 | +0.00 | 932.11 | +0.00 | 932.11 | +0.00 | 932.11 | +0.00 |
| No Eligible (%) | 26.21 % | 22.13 % | -4.08 pp | 26.04 % | -0.17 pp | 22.11 % | -4.10 pp | 26.10 % | -0.11 pp |
| IPC Active | 2.95 inst/cycle | 3.12 inst/cycle | +0.17 | 2.95 inst/cycle | +0.00 | 3.12 inst/cycle | +0.17 | 2.96 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0083× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | replace_tl_math_rsqrt_with_native_tl_rsqrt_to_reduce_instruction_count | 1.000× | 1.000× | ✓ |  |
| D | replace_explicit_exp_sigmoid_with_tl_sigmoid_builtin | 1.000× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_tl_math_r... | Δ | replace_explicit_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1220000 ns | 1220000 ns | +0 ns (+0.0%) | 1210000 ns | -10000 ns (-0.8%) |
| SM Throughput (%) | 73.77 % | 77.85 % | +4.08 pp | 73.75 % | -0.02 pp |
| Memory Throughput (%) | 33.43 % | 33.31 % | -0.12 pp | 33.45 % | +0.02 pp |
| DRAM Throughput (%) | 33.43 % | 33.31 % | -0.12 pp | 33.45 % | +0.02 pp |
| Achieved Occupancy (%) | 68.38 % | 72.63 % | +4.25 pp | 68.41 % | +0.03 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 932.11 | 932.11 | +0.00 | 932.11 | +0.00 |
| No Eligible (%) | 26.21 % | 22.15 % | -4.06 pp | 26.08 % | -0.13 pp |
| IPC Active | 2.95 inst/cycle | 3.12 inst/cycle | +0.17 | 2.95 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0083× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | enable_software_pipelining_via_num_stages_to_hide_l1tex_latency | 1.000× | 1.000× | ✓ |  |
| D | replace_tl_exp_with_libdevice_fast_expf_in_sigmoid | 1.001× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | enable_software_p... | Δ | replace_tl_exp_wi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1220000 ns | 1220000 ns | +0 ns (+0.0%) | 1210000 ns | -10000 ns (-0.8%) |
| SM Throughput (%) | 73.77 % | 77.78 % | +4.01 pp | 75.45 % | +1.68 pp |
| Memory Throughput (%) | 33.43 % | 33.33 % | -0.10 pp | 33.33 % | -0.10 pp |
| DRAM Throughput (%) | 33.43 % | 33.33 % | -0.10 pp | 33.33 % | -0.10 pp |
| Achieved Occupancy (%) | 68.38 % | 72.79 % | +4.41 pp | 66.35 % | -2.03 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 932.11 | 932.11 | +0.00 | 932.11 | +0.00 |
| No Eligible (%) | 26.21 % | 22.09 % | -4.12 pp | 24.50 % | -1.71 pp |
| IPC Active | 2.95 inst/cycle | 3.12 inst/cycle | +0.17 | 3.02 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  1220000 ns  (1.000×)
Iter 2     1210000 ns  (1.008×)  — replace_libdevice_rsqrt_and_reduce_nonfused_fp32_instructions

Final best: 1210000 ns  (1.0083×)
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
 def triton_poi_fused_convolution_hardswish_mul_native_group_norm_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
@@ -12,25 +13,26 @@

     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x5 = xindex
+    x5 = tl.max_contiguous(tl.multiple_of(x5, 4), 4)  # hint: enables 128-bit vectorized load
     x2 = ((xindex // 123039) % 16)
     x7 = xindex // 123039
     x0 = (xindex % 3969)
     x4 = xindex // 3969
-    tmp0 = tl.load(in_ptr0 + (x5), xmask)
+    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
     tmp5 = tl.load(in_ptr2 + (x7 // 4), xmask, eviction_policy='evict_last')
     tmp7 = tl.load(in_ptr3 + (x7 // 4), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tmp3 = tl.sigmoid(tmp2)
+    tmp3 = 1.0 / (1.0 + libdevice.fast_expf(-tmp2))  # sigmoid(x) -- MUFU.EX2, fast
     tmp4 = tmp3 * tmp2
     tmp6 = tmp4 - tmp5
     tmp8 = tl.full([1], 492156.0, tl.float32)
     tmp9 = (tmp7 / tmp8)
     tmp10 = tl.full([1], 1e-05, tl.float32)
     tmp11 = tmp9 + tmp10
-    tmp12 = libdevice.rsqrt(tmp11)
+    tmp12 = tl.rsqrt(tmp11)
     tmp13 = tmp6 * tmp12
     tmp15 = tmp13 * tmp14
     tmp17 = tmp15 + tmp16
@@ -44,8 +46,6 @@

     tmp25 = tl.full([1], 0.16666666666666666, tl.float32)
     tmp26 = tmp24 * tmp25
     tl.store(out_ptr1 + (x5), tmp26, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
