# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__softmax_convolution_exp_sub_40` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 79170 ns — **Best latency**: 52510 ns (**1.5077×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_to_eliminate_smem_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__softmax_convolution_exp_sub_40`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_40_triton_tem_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 3.10
- **Registers/thread**: 96
- **Duration**: 79170 ns
- **SM Throughput**: 20.59 % | **Memory**: 89.92 % | **DRAM**: 18.11 %
- **Achieved Occupancy**: 29.02 %
- **IPC Active**: 0.88 inst/cycle | **No Eligible**: 78.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_eliminate_smem_bank_conflicts | — | 1.508× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 79170 ns | 52510 ns | -26660 ns (-33.7%) |
| SM Throughput (%) | 20.59 % | 11.64 % | -8.95 pp |
| Memory Throughput (%) | 89.92 % | 64.44 % | -25.48 pp |
| DRAM Throughput (%) | 18.11 % | 27.36 % | +9.25 pp |
| Achieved Occupancy (%) | 29.02 % | 34.11 % | +5.09 pp |
| Registers/Thread | 96.00 register/thread | 39.00 register/thread | -57 |
| Waves/SM | 3.10 | 2.59 | -0.51 |
| No Eligible (%) | 78.10 % | 86.53 % | +8.43 pp |
| IPC Active | 0.88 inst/cycle | 0.54 inst/cycle | -0.34 |

---

## Iteration 2

**Best so far**: 1.5077× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_dot_with_register_accumulation_to_eliminate_smem_bank_conflicts | — | 0.780× | ✓ |  |
| B | preload_b_into_padded_shared_memory_to_eliminate_bank_conflicts | — | 1.534× | ✓ |  |
| C | increase_num_warps_to_hide_mio_latency_and_reduce_scheduler_starvation | — | 1.547× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_tl_dot_wi... | Δ | preload_b_into_pa... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 79170 ns | 101540 ns | +22370 ns (+28.3%) | 51620 ns | -27550 ns (-34.8%) | 51170 ns | -28000 ns (-35.4%) |
| SM Throughput (%) | 20.59 % | 35.72 % | +15.13 pp | 12.00 % | -8.59 pp | 11.67 % | -8.92 pp |
| Memory Throughput (%) | 89.92 % | 84.11 % | -5.81 pp | 66.34 % | -23.58 pp | 64.64 % | -25.28 pp |
| DRAM Throughput (%) | 18.11 % | 14.21 % | -3.90 pp | 27.83 % | +9.72 pp | 28.15 % | +10.04 pp |
| Achieved Occupancy (%) | 29.02 % | 12.06 % | -16.96 pp | 34.59 % | +5.57 pp | 34.23 % | +5.21 pp |
| Registers/Thread | 96.00 register/thread | 255.00 register/thread | +159 | 40.00 register/thread | -56 | 39.00 register/thread | -57 |
| Waves/SM | 3.10 | 7.76 | +4.66 | 2.59 | -0.51 | 2.59 | -0.51 |
| No Eligible (%) | 78.10 % | 79.84 % | +1.74 pp | 85.84 % | +7.74 pp | 85.94 % | +7.84 pp |
| IPC Active | 0.88 inst/cycle | 0.81 inst/cycle | -0.07 | 0.56 inst/cycle | -0.32 | 0.54 inst/cycle | -0.34 |


### Key Learnings

#### ✅ Plan A: `replace_tl_dot_with_register_accumulation_to_eliminate_smem_bank_conflicts` (1.533×)

When a kernel has been modified to replace tl.dot with a 3D expand-and-sum pattern that causes register spilling (registers/thread at hardware max of 255, theoretical occupancy collapsed to 12.5%, instruction count 3x higher), reverting to tl.dot restores correct register usage (39/thread), occupancy (37.5%), and instruction count (4.7M), cutting duration by ~50%. The only code change needed is replacing the 3D intermediate accumulation with tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE); the surrounding load logic and epilogue remain unchanged. This is a corrective revert, not a new optimization — the negative experience (neg_002) documents the anti-pattern and this positive confirms the fix.

> **Applicable when**: Kernel shows registers_per_thread at or near hardware maximum (255), theoretical occupancy collapsed to <=12.5%, instruction count 3x+ higher than expected, and UncoalescedSharedAccess/local memory NCU rules firing — all symptoms of a prior 3D expand-and-sum replacement of tl.dot having caused register spill.

> **Avoid when**: The kernel is already using tl.dot correctly (registers/thread ~39, occupancy ~37.5%); or when the 3D intermediate tensor is small enough to fit in registers (BLOCK_M * BLOCK_K * BLOCK_N < ~4K elements).

---

## Iteration 3

**Best so far**: 1.5077× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | revert_register_spill_by_restoring_tl_dot_for_memory_coalescing | — | 1.533× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_register_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 79170 ns | 51650 ns | -27520 ns (-34.8%) |
| SM Throughput (%) | 20.59 % | 11.67 % | -8.92 pp |
| Memory Throughput (%) | 89.92 % | 64.66 % | -25.26 pp |
| DRAM Throughput (%) | 18.11 % | 27.85 % | +9.74 pp |
| Achieved Occupancy (%) | 29.02 % | 34.33 % | +5.31 pp |
| Registers/Thread | 96.00 register/thread | 39.00 register/thread | -57 |
| Waves/SM | 3.10 | 2.59 | -0.51 |
| No Eligible (%) | 78.10 % | 86.57 % | +8.47 pp |
| IPC Active | 0.88 inst/cycle | 0.55 inst/cycle | -0.33 |

---

## Cumulative Progress

```
Baseline:  79170 ns  (1.000×)
Iter 1     52510 ns  (1.508×)  — autotune_tile_sizes_to_eliminate_smem_bank_conflicts

Final best: 52510 ns  (1.5077×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__softmax_convolution_exp_sub_40(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_N : tl.constexpr = 4
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -72,9 +73,7 @@

         xindex = idx_n + 4*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +87,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
