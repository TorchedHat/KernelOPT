# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 53090 ns — **Best latency**: 52670 ns (**1.0080×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tma_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_3_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 3.71
- **Registers/thread**: 214
- **Duration**: 53090 ns
- **SM Throughput**: 23.59 % | **Memory**: 59.65 % | **DRAM**: 8.73 %
- **Achieved Occupancy**: 12.41 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 72.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.997× | 1.002× | ✓ |  |
| D | reduce_register_pressure_via_tile_size_and_autotune | 1.173× | 1.179× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53090 ns | 52990 ns | -100 ns (-0.2%) | 45020 ns | -8070 ns (-15.2%) |
| SM Throughput (%) | 23.59 % | 23.61 % | +0.02 pp | 27.91 % | +4.32 pp |
| Memory Throughput (%) | 59.65 % | 59.71 % | +0.06 pp | 48.75 % | -10.90 pp |
| DRAM Throughput (%) | 8.73 % | 8.69 % | -0.04 pp | 11.08 % | +2.35 pp |
| Achieved Occupancy (%) | 12.41 % | 12.40 % | -0.01 pp | 11.79 % | -0.62 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 255.00 register/thread | +41 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 1.86 | -1.85 |
| No Eligible (%) | 72.82 % | 72.90 % | +0.08 pp | 68.81 % | -4.01 pp |
| IPC Active | 1.08 inst/cycle | 1.08 inst/cycle | +0.00 | 1.26 inst/cycle | +0.18 |

---

## Iteration 2

**Best so far**: 1.1793× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_index_to_eliminate_shared_memory_bank_conflicts | 1.175× | 1.184× | ✓ |  |
| B | tma_loads_hopper_tensor_memory_accelerator | 1.180× | 1.188× | ✓ |  |
| C | fix_b_matrix_load_access_pattern_to_eliminate_bank_conflicts | 1.170× | 1.005× | ✓ |  |
| D | tma_loads | 1.186× | 1.008× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.176× | 1.197× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_inde... | Δ | tma_loads_hopper_... | Δ | fix_b_matrix_load... | Δ | tma_loads | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53090 ns | 44830 ns | -8260 ns (-15.6%) | 44700 ns | -8390 ns (-15.8%) | 52830 ns | -260 ns (-0.5%) | 52670 ns | -420 ns (-0.8%) | 44350 ns | -8740 ns (-16.5%) |
| SM Throughput (%) | 23.59 % | 28.21 % | +4.62 pp | 27.57 % | +3.98 pp | 23.63 % | +0.04 pp | 23.58 % | -0.01 pp | 27.77 % | +4.18 pp |
| Memory Throughput (%) | 59.65 % | 49.26 % | -10.39 pp | 48.48 % | -11.17 pp | 59.76 % | +0.11 pp | 59.64 % | -0.01 pp | 48.51 % | -11.14 pp |
| DRAM Throughput (%) | 8.73 % | 11.13 % | +2.40 pp | 11.26 % | +2.53 pp | 8.73 % | +0.00 pp | 8.77 % | +0.04 pp | 11.40 % | +2.67 pp |
| Achieved Occupancy (%) | 12.41 % | 11.78 % | -0.63 pp | 11.81 % | -0.60 pp | 12.38 % | -0.03 pp | 12.39 % | -0.02 pp | 11.72 % | -0.69 pp |
| Registers/Thread | 214.00 register/thread | 255.00 register/thread | +41 | 255.00 register/thread | +41 | 214.00 register/thread | +0 | 214.00 register/thread | +0 | 255.00 register/thread | +41 |
| Waves/SM | 3.71 | 1.86 | -1.85 | 1.86 | -1.85 | 3.71 | +0.00 | 3.71 | +0.00 | 1.86 | -1.85 |
| No Eligible (%) | 72.82 % | 67.64 % | -5.18 pp | 68.29 % | -4.53 pp | 72.90 % | +0.08 pp | 72.99 % | +0.17 pp | 68.59 % | -4.23 pp |
| IPC Active | 1.08 inst/cycle | 1.25 inst/cycle | +0.17 | 1.25 inst/cycle | +0.17 | 1.08 inst/cycle | +0.00 | 1.07 inst/cycle | -0.01 | 1.26 inst/cycle | +0.18 |

---

## Iteration 3

**Best so far**: 1.1793× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_double_occupancy | 0.997× | 0.592× | ✓ |  |
| B | fix_b_block_ptr_order_to_eliminate_shared_memory_transpose_conflicts | 1.186× | 1.017× | ✓ |  |
| C | cap_register_usage_with_maxnreg_to_improve_occupancy | 0.801× | 0.588× | ✓ |  |
| D | cap_registers_per_thread_with_maxnreg_to_increase_occupancy | 1.171× | 1.176× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | fix_b_block_ptr_o... | Δ | cap_register_usag... | Δ | cap_registers_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 53090 ns | 89630 ns | +36540 ns (+68.8%) | 52220 ns | -870 ns (-1.6%) | 90270 ns | +37180 ns (+70.0%) | 45150 ns | -7940 ns (-15.0%) |
| SM Throughput (%) | 23.59 % | 21.88 % | -1.71 pp | 23.79 % | +0.20 pp | 21.49 % | -2.10 pp | 27.95 % | +4.36 pp |
| Memory Throughput (%) | 59.65 % | 48.18 % | -11.47 pp | 60.18 % | +0.53 pp | 47.37 % | -12.28 pp | 48.80 % | -10.85 pp |
| DRAM Throughput (%) | 8.73 % | 32.86 % | +24.13 pp | 8.82 % | +0.09 pp | 32.72 % | +23.99 pp | 11.14 % | +2.41 pp |
| Achieved Occupancy (%) | 12.41 % | 22.50 % | +10.09 pp | 12.37 % | -0.04 pp | 22.53 % | +10.12 pp | 11.79 % | -0.62 pp |
| Registers/Thread | 214.00 register/thread | 128.00 register/thread | -86 | 214.00 register/thread | +0 | 128.00 register/thread | -86 | 255.00 register/thread | +41 |
| Waves/SM | 3.71 | 0.93 | -2.78 | 3.71 | +0.00 | 0.93 | -2.78 | 1.86 | -1.85 |
| No Eligible (%) | 72.82 % | 80.20 % | +7.38 pp | 72.97 % | +0.15 pp | 80.19 % | +7.37 pp | 68.20 % | -4.62 pp |
| IPC Active | 1.08 inst/cycle | 0.78 inst/cycle | -0.30 | 1.08 inst/cycle | +0.00 | 0.78 inst/cycle | -0.30 | 1.25 inst/cycle | +0.17 |


### Key Learnings

#### ❌ Plan A: `cap_registers_with_maxnreg_to_double_occupancy` (0.592×)

Capping registers via maxnreg in Triton autotune configs successfully doubled theoretical occupancy (12.5% → 25%) and achieved occupancy (11.72% → 22.5%), but caused a severe 1.69x regression in kernel duration (44.35 µs → 89.63 µs). The register spilling triggered by maxnreg=128 collapsed L1 hit rate from 17.55% to 2.36% and tripled DRAM throughput (11.4% → 32.86%), while warp cycles per issued instruction exploded from 6.04 to 18.26 — indicating the spill traffic to local memory created a far worse L1TEX stall bottleneck (CPIStall: 68% of cycles stalled on L1TEX, 12.4 cycles average) than the original low-occupancy latency exposure. The additional warps from higher occupancy could not hide the dramatically increased memory latency from register spills, making the net effect strongly negative.

> **Applicable when**: Never apply maxnreg to kernels that already use near-maximum registers (255/thread) when the kernel body cannot be restructured to reduce register demand. The spill traffic will always exceed the latency-hiding benefit of higher occupancy when the kernel's register footprint is intrinsically large.

> **Avoid when**: When the kernel uses 255 registers/thread (hardware maximum) and the NCU MemoryCacheAccessPattern rule already shows local load/store inefficiency (only 1.0 of 32 bytes utilized per sector) — this is a pre-existing spill signal. Adding maxnreg will dramatically worsen spilling: L1 hit rate will collapse (17% → 2%), warp cycles per instruction will triple (6 → 18), and DRAM throughput will spike. The occupancy gain (12.5% → 25%) cannot compensate for 3x worse memory stall cycles. Do not apply when the kernel's register pressure is driven by a large accumulator tile (e.g., BLOCK_M × BLOCK_N output elements per thread) that cannot be reduced without changing tile sizes.

#### ❌ Plan C: `cap_register_usage_with_maxnreg_to_improve_occupancy` (0.588×)

Capping registers via maxnreg in Triton autotune configs successfully doubled theoretical occupancy (12.5% → 25%) and block limit registers (4 → 8 blocks/SM), but caused catastrophic register spilling to local memory. The spill traffic collapsed L1 hit rate from 17.55% to 2.41%, increased DRAM throughput from 11.4% to 32.72%, inflated executed instructions by ~30% (9.7M → 12.7M), and caused warp cycles per issued instruction to explode from 6.04 to 18.38 — a 3x increase in per-instruction stall depth. The net result was a 1.7x slowdown (44.35 µs → 90.27 µs). When a kernel uses 255 registers/thread due to complex computation (not just tile size), forcing maxnreg causes the compiler to spill heavily to local memory with strided access patterns, which is far more costly than the occupancy gain.

> **Applicable when**: NEVER apply maxnreg caps when the kernel's register usage is driven by computational complexity (many live values from fused operations like batch_norm + conv + relu) rather than tile size alone. The anti-pattern is: maxnreg cap → compiler spills registers to local memory → strided local load/store traffic → L1 hit rate collapses → warp stall cycles explode → net slowdown despite higher occupancy.

> **Avoid when**: Apply maxnreg only when: (1) register pressure is dominated by tile-size-proportional storage (e.g., large accumulator tiles) rather than complex fused computation, AND (2) the kernel has simple enough logic that the compiler can spill cheaply (e.g., loop-invariant values). Do not apply when NCU reports MemoryCacheAccessPattern with 1.0/32 bytes utilized per sector for local loads — this indicates existing spill is already strided and adding more spill will compound the problem.

---

## Iteration 4

**Best so far**: 1.1793× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | bypass_shared_memory_staging_for_b_matrix_via_direct_register_load | 1.171× | 1.021× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_block_ptr_swizzle_order | 0.812× | 0.579× | ✓ |  |


### Metric Comparison

| Metric | Baseline | bypass_shared_mem... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53090 ns | 52000 ns | -1090 ns (-2.1%) | 91710 ns | +38620 ns (+72.7%) |
| SM Throughput (%) | 23.59 % | 24.02 % | +0.43 pp | 21.24 % | -2.35 pp |
| Memory Throughput (%) | 59.65 % | 60.77 % | +1.12 pp | 46.72 % | -12.93 pp |
| DRAM Throughput (%) | 8.73 % | 8.82 % | +0.09 pp | 32.16 % | +23.43 pp |
| Achieved Occupancy (%) | 12.41 % | 12.37 % | -0.04 pp | 22.52 % | +10.11 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 128.00 register/thread | -86 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 0.93 | -2.78 |
| No Eligible (%) | 72.82 % | 72.97 % | +0.15 pp | 80.67 % | +7.85 pp |
| IPC Active | 1.08 inst/cycle | 1.08 inst/cycle | +0.00 | 0.76 inst/cycle | -0.32 |

---

## Cumulative Progress

```
Baseline:  53090 ns  (1.000×)
Iter 1     45020 ns  (1.179×)  — reduce_register_pressure_via_tile_size_and_autotune

Final best: 52670 ns  (1.0080×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original 8 configs unchanged
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        # maxnreg=128: 128*64=8192 regs/block -> 8 blocks/SM -> 25% occupancy
+        triton.Config({}, num_warps=2, num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=4, num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=8, num_stages=4, maxnreg=128),
+        # maxnreg=96: 96*64=6144 regs/block -> ~10 blocks/SM -> ~31% occupancy
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=4, num_stages=4, maxnreg=96),
+        # maxnreg=64: 64*64=4096 regs/block -> 16 blocks/SM -> 50% occupancy
+        triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=4, num_stages=1, maxnreg=64),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_3(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -67,10 +96,12 @@

         xindex = idx_n + 32*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 64*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Replace broadcast_to-based B index with explicit stride-correct index
+        # k_offs * 1 + n_offs * 32 is mathematically identical to idx_m + 32*idx_n
+        # but expresses stride_bk=1 and stride_bn=32 explicitly, enabling coalesced loads
+        k_offs = offs_k[:, None] + k_idx * BLOCK_K   # [BLOCK_K, 1]
+        n_offs = offs_b_n[None, :]                    # [1, BLOCK_N]
+        b = tl.load(B + k_offs * 1 + n_offs * 32)    # [BLOCK_K, BLOCK_N], stride_bk=1 stride_bn=32
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +117,6 @@

     # inductor generates a suffix
     xindex = idx_n + 64*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
