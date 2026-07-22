# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_view_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 90370 ns — **Best latency**: 90400 ns (**0.9997×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_view_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_10_triton_tem_fused_addmm_t_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (112,1,1), Waves/SM: 0.28
- **Registers/thread**: 168
- **Duration**: 90370 ns
- **SM Throughput**: 23.21 % | **Memory**: 52.35 % | **DRAM**: 1.14 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 69.92 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.000× | ✓ | **✓ best** |
| B | add_autotune_reduce_register_pressure_increase_occupancy | 1.000× | 0.998× | ✓ |  |
| C | eviction_policy_streaming_loads | 0.999× | 0.997× | ✓ |  |
| D | add_autotune_reduce_register_pressure_improve_occupancy | 0.997× | 0.547× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ | eviction_policy_s... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 90370 ns | 90400 ns | +30 ns (+0.0%) | 90530 ns | +160 ns (+0.2%) | 90620 ns | +250 ns (+0.3%) | 165310 ns | +74940 ns (+82.9%) |
| SM Throughput (%) | 23.21 % | 23.21 % | +0.00 pp | 23.17 % | -0.04 pp | 23.19 % | -0.02 pp | 12.74 % | -10.47 pp |
| Memory Throughput (%) | 52.35 % | 52.35 % | +0.00 pp | 52.26 % | -0.09 pp | 52.30 % | -0.05 pp | 18.73 % | -33.62 pp |
| DRAM Throughput (%) | 1.14 % | 1.14 % | +0.00 pp | 1.14 % | +0.00 pp | 1.13 % | -0.01 pp | 0.63 % | -0.51 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 255.00 register/thread | +87 |
| Waves/SM | 0.28 | 0.28 | +0.00 | 0.28 | +0.00 | 0.28 | +0.00 | 0.21 | -0.07 |
| No Eligible (%) | 69.92 % | 69.89 % | -0.03 pp | 69.86 % | -0.06 pp | 69.94 % | +0.02 pp | 66.35 % | -3.57 pp |
| IPC Active | 1.20 inst/cycle | 1.20 inst/cycle | +0.00 | 1.20 inst/cycle | +0.00 | 1.20 inst/cycle | +0.00 | 0.67 inst/cycle | -0.53 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts | 0.998× | 0.973× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 90370 ns | 92900 ns | +2530 ns (+2.8%) |
| SM Throughput (%) | 23.21 % | 22.39 % | -0.82 pp |
| Memory Throughput (%) | 52.35 % | 50.81 % | -1.54 pp |
| DRAM Throughput (%) | 1.14 % | 1.11 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.28 | 0.28 | +0.00 |
| No Eligible (%) | 69.92 % | 72.30 % | +2.38 pp |
| IPC Active | 1.20 inst/cycle | 1.11 inst/cycle | -0.09 |


### Key Learnings

#### ⚠️ Plan C: `autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts` (0.000×)

Plan 'autotune_tile_sizes_to_reduce_register_pressure_and_bank_conflicts' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=1.204e+02 > atol=0.001
seed=42 output[0]: max_diff=1.118e+02 > atol=0.001
seed=1337 output[0]: max_diff=1.188e+02 > atol=0.001

> **Applicable when**: The kernel has BLOCK_M=64, BLOCK_N=128, BLOCK_K=32 hardcoded as tl.constexpr inside the kernel body. The existing @triton.autotune only varies num_warps/num_stages via empty {} config dicts, which cannot change tile sizes. This causes 168 registers/thread (register-limited: Block Limit Registers=3 is the binding constraint), yielding only 6.25% achieved occupancy on H200. The large BLOCK_M*BLOCK_N=8192 accumulator tile is the primary driver of register pressure. Additionally, 8-way shared memory bank conflicts (69.78% of wavefronts) stall warps for 47% of cycles. Smaller tiles (e.g., BLOCK_M=32, BLOCK_N=64) will reduce the accumulator size from 8192 to 2048 elements, dramatically cutting register usage and enabling more blocks per SM. Memory item exp_001 confirms that autotune without tile size variation regressed performance by 83%; the fix is to expose BLOCK_M, BLOCK_N, BLOCK_K as tl.constexpr kernel arguments and include them in config dicts.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=1.204e+02 > atol=0.001
seed=42 output[0]: max_diff=1.118e+02 > atol=0.001
seed=1337 output[0]: max_diff=1.188e+02 > atol=0.001

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads | 0.993× | 0.987× | ✓ |  |
| C | cap_registers_with_maxnreg_to_improve_occupancy | 0.823× | 0.780× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 90370 ns | 91520 ns | +1150 ns (+1.3%) | 115840 ns | +25470 ns (+28.2%) |
| SM Throughput (%) | 23.21 % | 22.71 % | -0.50 pp | 20.46 % | -2.75 pp |
| Memory Throughput (%) | 52.35 % | 51.26 % | -1.09 pp | 43.67 % | -8.68 pp |
| DRAM Throughput (%) | 1.14 % | 1.12 % | -0.02 pp | 0.89 % | -0.25 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 165.00 register/thread | -3 | 128.00 register/thread | -40 |
| Waves/SM | 0.28 | 0.28 | +0.00 | 0.21 | -0.07 |
| No Eligible (%) | 69.92 % | 71.36 % | +1.44 pp | 73.42 % | +3.50 pp |
| IPC Active | 1.20 inst/cycle | 1.15 inst/cycle | -0.05 | 1.07 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan C: `cap_registers_with_maxnreg_to_improve_occupancy` (0.780×)

Adding maxnreg to Triton configs reduced registers/thread from 168 to 128 and raised Block Limit Registers from 3 to 4, but achieved occupancy remained stuck at 6.25% — the grid (112 blocks < 132 SMs) was the binding constraint, not register count. The register cap forced the compiler to spill to local memory, introducing uncoalesced local load/store traffic (new MemoryCacheAccessPattern rule, 42.3% speedup potential flagged) and increasing executed instructions by ~15% (16.4M → 18.9M), causing a 24% slowdown. When the kernel is wave-limited (grid_size < sm_count) rather than occupancy-limited, capping registers via maxnreg cannot improve throughput and instead adds spill overhead that degrades performance.

> **Applicable when**: Never effective when grid_size < sm_count (wave-limited regime). Only potentially useful when: (1) achieved occupancy is genuinely register-limited (Block Limit Registers is the binding constraint AND grid_size >> sm_count so more blocks can actually run), AND (2) the register spill cost to L2/shared is lower than the occupancy gain benefit.

> **Avoid when**: Grid is wave-limited (grid_size <= sm_count): capping registers cannot increase the number of concurrently running blocks because there are not enough blocks to fill the GPU regardless. Also avoid when the kernel already has significant shared memory bank conflicts — spilled registers go to local memory via L1TEX and add uncoalesced access patterns on top of existing bottlenecks. Profiling signals: waves_per_sm < 1.0 AND grid_size < sm_count → do not apply maxnreg.

---

## Cumulative Progress

```
Baseline:  90370 ns  (1.000×)

Final best: 90400 ns  (0.9997×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_t_view_10(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,12 +79,9 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 512*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 2048*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + (b_k_idx_vals + 512*offs_b_n[None, :]), eviction_policy='evict_first')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -86,8 +97,6 @@

     # inductor generates a suffix
     xindex = idx_n + 2048*idx_m
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
