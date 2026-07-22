# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 979010 ns — **Best latency**: 995620 ns (**0.9833×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_038/kernel_1_triton_red_fused__softmax_avg_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 1.94
- **Registers/thread**: 64
- **Duration**: 979010 ns
- **SM Throughput**: 51.07 % | **Memory**: 66.07 % | **DRAM**: 66.07 %
- **Achieved Occupancy**: 44.43 %
- **IPC Active**: 2.11 inst/cycle | **No Eligible**: 46.98 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | memory_coalescing_via_xblock_autotune | 1.001× | 1.002× | ✓ | **✓ best** |
| B | increase_xblock_with_autotune_to_eliminate_partial_wave | 1.000× | 0.999× | ✓ |  |
| C | increase_xblock_and_add_autotune_to_eliminate_partial_wave | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | increase_xblock_w... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 979010 ns | 977280 ns | -1730 ns (-0.2%) | 980320 ns | +1310 ns (+0.1%) | 979300 ns | +290 ns (+0.0%) |
| SM Throughput (%) | 51.07 % | 51.19 % | +0.12 pp | 50.94 % | -0.13 pp | 50.88 % | -0.19 pp |
| Memory Throughput (%) | 66.07 % | 66.20 % | +0.13 pp | 65.98 % | -0.09 pp | 66.05 % | -0.02 pp |
| DRAM Throughput (%) | 66.07 % | 66.20 % | +0.13 pp | 65.98 % | -0.09 pp | 66.05 % | -0.02 pp |
| Achieved Occupancy (%) | 44.43 % | 44.43 % | +0.00 pp | 44.44 % | +0.01 pp | 44.50 % | +0.07 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 46.98 % | 47.33 % | +0.35 pp | 47.27 % | +0.29 pp | 47.27 % | +0.29 pp |
| IPC Active | 2.11 inst/cycle | 2.11 inst/cycle | +0.00 | 2.11 inst/cycle | +0.00 | 2.10 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0018× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_xblock_and_add_register_reducing_autotune_configs | 1.000× | 1.001× | ✓ |  |
| B | reduce_register_pressure_to_improve_occupancy_and_hide_l1tex_stalls | 1.048× | 0.648× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 1.070× | 0.951× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_xblock_a... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 979010 ns | 978050 ns | -960 ns (-0.1%) | 1510000 ns | +530990 ns (+54.2%) | 1030000 ns | +50990 ns (+5.2%) |
| SM Throughput (%) | 51.07 % | 51.04 % | -0.03 pp | 36.34 % | -14.73 pp | 52.51 % | +1.44 pp |
| Memory Throughput (%) | 66.07 % | 66.13 % | +0.06 pp | 62.95 % | -3.12 pp | 64.20 % | -1.87 pp |
| DRAM Throughput (%) | 66.07 % | 66.13 % | +0.06 pp | 62.95 % | -3.12 pp | 64.20 % | -1.87 pp |
| Achieved Occupancy (%) | 44.43 % | 44.51 % | +0.08 pp | 82.77 % | +38.34 pp | 50.74 % | +6.31 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 32.00 register/thread | -32 | 48.00 register/thread | -16 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 0.97 | -0.97 | 1.55 | -0.39 |
| No Eligible (%) | 46.98 % | 47.32 % | +0.34 pp | 60.37 % | +13.39 pp | 46.77 % | -0.21 pp |
| IPC Active | 2.11 inst/cycle | 2.11 inst/cycle | +0.00 | 1.58 inst/cycle | -0.53 | 2.13 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan C: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (0.648×)

Capping registers via maxnreg=32 doubled theoretical occupancy (50%→100%) and achieved occupancy (44%→83%), but caused severe register spilling to local memory. The NCU report shows new 'local loads/stores' cache access pattern warnings with only 1.0/32 bytes utilized per sector, and Warp Cycles Per Issued Instruction nearly tripled (13.49→33.21 cycles). The L1TEX stall fraction remained dominant (71.8%) but the stall duration per warp increased from 9.5 to 23.8 cycles, indicating spill traffic overwhelmed any latency-hiding benefit from higher occupancy. More eligible warps (1.08→0.67 actually dropped) could not compensate for the additional memory traffic from spills.

> **Applicable when**: Never: this specific strategy (maxnreg=32 on a kernel with 64 registers/thread doing large reduction loops) caused regression. Avoid when the kernel has complex per-element computation requiring many live values across a long reduction loop.

> **Avoid when**: Kernel has high register count (≥48) due to complex arithmetic in a long reduction loop (R0_BLOCK≥512). In such cases, maxnreg forces spilling to local memory, which generates strided 1-byte-per-32-byte-sector local load/store traffic that is far more expensive than the original L1TEX global load stalls. The resulting Warp Cycles Per Issued Instruction will increase dramatically (e.g., 13→33 cycles) despite higher occupancy.

---

## Iteration 3

**Best so far**: 1.0018× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_eliminate_partial_wave_tail | 1.000× | 1.000× | ✓ |  |
| B | remove_maxnreg_cap_to_eliminate_register_spill_local_store_traffic | 1.159× | 0.983× | ✓ | **✓ best** |
| C | eliminate_register_spill_by_removing_maxnreg_cap | 1.065× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | remove_maxnreg_ca... | Δ | eliminate_registe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 979010 ns | 978560 ns | -450 ns (-0.0%) | 995620 ns | +16610 ns (+1.7%) | 1000000 ns | +20990 ns (+2.1%) |
| SM Throughput (%) | 51.07 % | 50.87 % | -0.20 pp | 50.00 % | -1.07 pp | 50.05 % | -1.02 pp |
| Memory Throughput (%) | 66.07 % | 66.09 % | +0.02 pp | 65.23 % | -0.84 pp | 64.79 % | -1.28 pp |
| DRAM Throughput (%) | 66.07 % | 66.09 % | +0.02 pp | 65.23 % | -0.84 pp | 64.79 % | -1.28 pp |
| Achieved Occupancy (%) | 44.43 % | 44.38 % | -0.05 pp | 44.23 % | -0.20 pp | 44.29 % | -0.14 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 46.98 % | 47.28 % | +0.30 pp | 47.51 % | +0.53 pp | 48.08 % | +1.10 pp |
| IPC Active | 2.11 inst/cycle | 2.11 inst/cycle | +0.00 | 2.07 inst/cycle | -0.04 | 2.07 inst/cycle | -0.04 |

---

## Iteration 4

**Best so far**: 1.0018× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_eviction_to_improve_l2_hit_rate_on_second_pass | 0.996× | 0.995× | ✓ |  |
| B | eliminate_double_global_load_via_num_stages_pipelining | 1.159× | 0.981× | ✓ |  |
| C | eliminate_second_pass_global_load_via_shared_memory_cache | 1.065× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_e... | Δ | eliminate_double_... | Δ | eliminate_second_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 979010 ns | 984450 ns | +5440 ns (+0.6%) | 997790 ns | +18780 ns (+1.9%) | 1000000 ns | +20990 ns (+2.1%) |
| SM Throughput (%) | 51.07 % | 50.70 % | -0.37 pp | 49.91 % | -1.16 pp | 50.04 % | -1.03 pp |
| Memory Throughput (%) | 66.07 % | 65.68 % | -0.39 pp | 65.09 % | -0.98 pp | 64.82 % | -1.25 pp |
| DRAM Throughput (%) | 66.07 % | 65.68 % | -0.39 pp | 65.09 % | -0.98 pp | 64.82 % | -1.25 pp |
| Achieved Occupancy (%) | 44.43 % | 44.45 % | +0.02 pp | 44.25 % | -0.18 pp | 44.24 % | -0.19 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 46.98 % | 47.35 % | +0.37 pp | 48.11 % | +1.13 pp | 48.22 % | +1.24 pp |
| IPC Active | 2.11 inst/cycle | 2.10 inst/cycle | -0.01 | 2.07 inst/cycle | -0.04 | 2.07 inst/cycle | -0.04 |

---

## Iteration 5

**Best so far**: 1.0018× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fma_fp32_instructions_in_clamp_exp_div_chain | 0.999× | 0.999× | ✓ |  |
| B | cache_first_pass_in_shared_memory_to_eliminate_second_global_load | 1.159× | 0.981× | ✓ |  |
| C | fuse_nonfused_fp32_instructions_via_fma_and_reciprocal | 1.067× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fma_fp32... | Δ | cache_first_pass_... | Δ | fuse_nonfused_fp3... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 979010 ns | 980190 ns | +1180 ns (+0.1%) | 997540 ns | +18530 ns (+1.9%) | 989630 ns | +10620 ns (+1.1%) |
| SM Throughput (%) | 51.07 % | 48.23 % | -2.84 pp | 49.78 % | -1.29 pp | 46.62 % | -4.45 pp |
| Memory Throughput (%) | 66.07 % | 65.96 % | -0.11 pp | 65.11 % | -0.96 pp | 65.63 % | -0.44 pp |
| DRAM Throughput (%) | 66.07 % | 65.96 % | -0.11 pp | 65.11 % | -0.96 pp | 65.63 % | -0.44 pp |
| Achieved Occupancy (%) | 44.43 % | 44.48 % | +0.05 pp | 44.22 % | -0.21 pp | 44.31 % | -0.12 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 46.98 % | 49.93 % | +2.95 pp | 48.14 % | +1.16 pp | 51.60 % | +4.62 pp |
| IPC Active | 2.11 inst/cycle | 2.01 inst/cycle | -0.10 | 2.07 inst/cycle | -0.04 | 1.94 inst/cycle | -0.17 |

---

## Cumulative Progress

```
Baseline:  979010 ns  (1.000×)
Iter 1     977280 ns  (1.002×)  — memory_coalescing_via_xblock_autotune

Final best: 995620 ns  (0.9833×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256},  num_warps=2,  num_stages=4),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
+        # New wave-aligned configs
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),  # grid=1024, fits in 1 wave
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),  # grid=1024, pipelined
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=4,  num_stages=3),  # grid=1024, deep pipeline
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),  # grid=512
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),  # grid=512, pipelined
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),  # grid=256
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),  # grid=256, pipelined
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -63,9 +93,7 @@

         tmp17 = tl.exp(tmp16)
         tmp18 = (tmp17 / tmp9)
         tmp20 = tmp18 * tmp19
-        tl.store(out_ptr2 + (r0_2 + 131072*x5), tmp20, xmask)
-
-
+        tl.store(out_ptr2 + (r0_2 + 131072*x5), tmp20, xmask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +103,7 @@

     xnumel = 2048
     r0_numel = 131072
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1[grid](arg_0, arg_1, arg_2, arg_3, 2048, 131072, XBLOCK=1, R0_BLOCK=2048)
+    triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1[grid](arg_0, arg_1, arg_2, arg_3, 2048, 131072)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
