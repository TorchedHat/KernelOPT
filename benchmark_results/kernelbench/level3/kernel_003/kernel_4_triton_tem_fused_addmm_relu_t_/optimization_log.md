# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 944260 ns — **Best latency**: 942880 ns (**1.0015×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tma_loads_with_autotune_to_reduce_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_4_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 944260 ns
- **SM Throughput**: 40.36 % | **Memory**: 58.15 % | **DRAM**: 1.32 %
- **Achieved Occupancy**: 12.29 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.002× | ✓ |  |
| B | tma_loads_with_autotune_to_reduce_register_pressure | 1.007× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 944260 ns | 942180 ns | -2080 ns (-0.2%) | 942880 ns | -1380 ns (-0.1%) |
| SM Throughput (%) | 40.36 % | 40.39 % | +0.03 pp | 40.37 % | +0.01 pp |
| Memory Throughput (%) | 58.15 % | 58.19 % | +0.04 pp | 58.05 % | -0.10 pp |
| DRAM Throughput (%) | 1.32 % | 1.33 % | +0.01 pp | 1.31 % | -0.01 pp |
| Achieved Occupancy (%) | 12.29 % | 12.29 % | +0.00 pp | 12.22 % | -0.07 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 58.43 % | 58.42 % | -0.01 pp | 58.54 % | +0.11 pp |
| IPC Active | 1.66 inst/cycle | 1.66 inst/cycle | +0.00 | 1.66 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0015× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_via_autotune_to_improve_occupancy | 0.875× | 0.286× | ✓ |  |
| B | tma_loads | 0.998× | 0.908× | ✓ |  |
| C | cap_registers_with_maxnreg_to_improve_occupancy | 1.000× | 0.141× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | tma_loads | Δ | cap_registers_wit... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 944260 ns | 3300000 ns | +2355740 ns (+249.5%) | 1040000 ns | +95740 ns (+10.1%) | 6690000 ns | +5745740 ns (+608.5%) |
| SM Throughput (%) | 40.36 % | 21.46 % | -18.90 pp | 36.48 % | -3.88 pp | 10.33 % | -30.03 pp |
| Memory Throughput (%) | 58.15 % | 39.52 % | -18.63 pp | 82.36 % | +24.21 pp | 19.79 % | -38.36 pp |
| DRAM Throughput (%) | 1.32 % | 36.90 % | +35.58 pp | 1.11 % | -0.21 pp | 16.38 % | +15.06 pp |
| Achieved Occupancy (%) | 12.29 % | 24.06 % | +11.77 pp | 12.50 % | +0.21 pp | 12.13 % | -0.16 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 232.00 register/thread | -23 | 128.00 register/thread | -127 |
| Waves/SM | 1.94 | 0.97 | -0.97 | 3.88 | +1.94 | 1.94 | +0.00 |
| No Eligible (%) | 58.43 % | 82.65 % | +24.22 pp | 62.13 % | +3.70 pp | 91.75 % | +33.32 pp |
| IPC Active | 1.66 inst/cycle | 0.70 inst/cycle | -0.96 | 1.51 inst/cycle | -0.15 | 0.33 inst/cycle | -1.33 |


### Key Learnings

#### ❌ Plan A: `cap_registers_with_maxnreg_via_autotune_to_improve_occupancy` (0.286×)

Applying maxnreg=128 via @triton.autotune successfully halved registers/thread (255→128) and doubled theoretical occupancy (12.5%→25%), but caused a 3.5x slowdown (942µs→3.3ms). The key regression signal is that warp cycles per issued instruction exploded from 4.72 to 22.13 cycles, and the no-eligible-warp percentage worsened from 58.4% to 82.65%. The CPIStall rule (new after the change) shows 71% of stall cycles are L1TEX scoreboard stalls, indicating that capping registers forced the compiler to spill accumulators to local memory, generating massive uncoalesced local load/store traffic (L2 bytes sent to compression unit grew 17x: 268MB→4.6GB). The extra memory traffic overwhelmed any occupancy benefit.

> **Applicable when**: Never: this strategy is a documented anti-pattern for large-tile GEMM kernels where the accumulator tile itself requires more registers than the maxnreg cap allows.

> **Avoid when**: Do NOT apply maxnreg capping when the kernel has a large accumulator tile (e.g., BLOCK_M x BLOCK_N = 128x128 fp32 = 16384 values requiring ~16384 registers just for the accumulator). Capping registers below what the accumulator needs forces spilling to local memory, causing L1TEX scoreboard stalls that are far more expensive than the low-occupancy latency being avoided. Warp cycles per issued instruction will increase dramatically (4.72→22.13 in this case) and kernel duration will worsen by 3-4x. The correct fix for register-limited occupancy in large-tile GEMM is to reduce BLOCK_M/BLOCK_N tile sizes, not to cap registers.

#### ❌ Plan C: `cap_registers_with_maxnreg_to_improve_occupancy` (0.141×)

Capping registers via maxnreg in Triton autotune configs did not improve occupancy (remained at 12.5%) because the shared memory limit (Block Limit Shared Mem=2) became the binding constraint after the compiler spilled excess registers to shared memory, increasing dynamic shared memory from 65 KB to 98 KB per block. The spilling caused Warp Cycles Per Issued Instruction to jump from 4.72 to 23.56 cycles (5x worse), eligible warps per scheduler collapsed from 0.76 to 0.09, and no-eligible-warp cycles rose from 58% to 92%, resulting in a 7x kernel slowdown. When a kernel is already shared-memory-limited (Block Limit Shared Mem <= Block Limit Registers), forcing register spills via maxnreg will overflow into shared memory, tighten the shared memory bottleneck, and dramatically worsen latency stalls rather than improving occupancy.

> **Applicable when**: Never applicable when shared memory is already a binding occupancy limiter (Block Limit Shared Mem <= Block Limit Registers). Only consider maxnreg when registers are the sole binding limiter AND there is ample shared memory headroom to absorb spills without tightening the shared memory block limit.

> **Avoid when**: Block Limit Shared Mem <= Block Limit Registers (shared memory is already co-limiting or more limiting than registers). In this case, register spills go to shared memory, increase dynamic smem per block, and make the shared memory constraint tighter, leaving occupancy unchanged while adding severe L1TEX stall latency (CPIStall rule triggered, Warp Cycles Per Issued Instruction increases 5x+).

---

## Iteration 3

**Best so far**: 1.0015× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_to_reduce_shared_memory_bank_conflicts | 0.959× | 0.959× | ✓ |  |
| B | remove_maxnreg_and_add_smaller_tile_autotune_configs | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | remove_maxnreg_an... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 944260 ns | 984900 ns | +40640 ns (+4.3%) | 944030 ns | -230 ns (-0.0%) |
| SM Throughput (%) | 40.36 % | 38.46 % | -1.90 pp | 40.42 % | +0.06 pp |
| Memory Throughput (%) | 58.15 % | 87.60 % | +29.45 pp | 58.23 % | +0.08 pp |
| DRAM Throughput (%) | 1.32 % | 1.17 % | -0.15 pp | 1.33 % | +0.01 pp |
| Achieved Occupancy (%) | 12.29 % | 12.50 % | +0.21 pp | 12.26 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 246.00 register/thread | -9 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 1.94 | +0.00 |
| No Eligible (%) | 58.43 % | 60.15 % | +1.72 pp | 58.58 % | +0.15 pp |
| IPC Active | 1.66 inst/cycle | 1.60 inst/cycle | -0.06 | 1.66 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  944260 ns  (1.000×)
Iter 1     942880 ns  (1.002×)  — tma_loads_with_autotune_to_reduce_register_pressure

Final best: 942880 ns  (1.0015×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=16, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_relu_t_4(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
@@ -91,7 +105,6 @@

     )
     tl.store(out_block_ptr, acc, boundary_check=(0, 1))
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
