# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_add_addmm_mul_pow_t_tanh_view_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16740000 ns — **Best latency**: 16340000 ns (**1.0245×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_add_addmm_mul_pow_t_tanh_view_13`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_13_triton_tem_fused_add_addmm_mul/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3072,1,1), Waves/SM: 11.64
- **Registers/thread**: 255
- **Duration**: 16740000 ns
- **SM Throughput**: 40.72 % | **Memory**: 59.02 % | **DRAM**: 1.49 %
- **Achieved Occupancy**: 12.34 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.23 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.017× | 1.024× | ✓ | **✓ best** |
| B | tma_loads_with_block_ptr_and_reduced_tile_sizes | 1.008× | 1.008× | ✓ |  |
| C | add_eviction_policies_for_streaming_loads | 1.000× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ | add_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16740000 ns | 16340000 ns | -400000 ns (-2.4%) | 16600000 ns | -140000 ns (-0.8%) | 16710000 ns | -30000 ns (-0.2%) |
| SM Throughput (%) | 40.72 % | 41.69 % | +0.97 pp | 40.96 % | +0.24 pp | 40.72 % | +0.00 pp |
| Memory Throughput (%) | 59.02 % | 60.19 % | +1.17 pp | 58.66 % | -0.36 pp | 59.01 % | -0.01 pp |
| DRAM Throughput (%) | 1.49 % | 1.55 % | +0.06 pp | 1.48 % | -0.01 pp | 1.49 % | +0.00 pp |
| Achieved Occupancy (%) | 12.34 % | 12.34 % | +0.00 pp | 12.48 % | +0.14 pp | 12.36 % | +0.02 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 11.64 | +0.00 | 11.64 | +0.00 | 11.64 | +0.00 |
| No Eligible (%) | 58.23 % | 57.11 % | -1.12 pp | 57.67 % | -0.56 pp | 58.20 % | -0.03 pp |
| IPC Active | 1.67 inst/cycle | 1.70 inst/cycle | +0.03 | 1.70 inst/cycle | +0.03 | 1.67 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0245× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_improve_occupancy | 0.882× | 0.268× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16740000 ns | 62370000 ns | +45630000 ns (+272.6%) |
| SM Throughput (%) | 40.72 % | 19.66 % | -21.06 pp |
| Memory Throughput (%) | 59.02 % | 37.19 % | -21.83 pp |
| DRAM Throughput (%) | 1.49 % | 37.19 % | +35.70 pp |
| Achieved Occupancy (%) | 12.34 % | 24.49 % | +12.15 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 |
| Waves/SM | 11.64 | 5.82 | -5.82 |
| No Eligible (%) | 58.23 % | 84.82 % | +26.59 pp |
| IPC Active | 1.67 inst/cycle | 0.61 inst/cycle | -1.06 |

---

## Iteration 3

**Best so far**: 1.0245× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr | 0.903× | 0.280× | ✓ |  |
| C | remove_maxnreg_to_eliminate_register_spilling | 1.015× | 1.017× | ✓ |  |
| D | increase_block_k_to_improve_arithmetic_intensity_and_reduce_l1tex_stalls | 0.809× | 0.201× | ✓ |  |
| E | add_triton_autotune | 1.014× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | remove_maxnreg_to... | Δ | increase_block_k_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16740000 ns | 59860000 ns | +43120000 ns (+257.6%) | 16460000 ns | -280000 ns (-1.7%) | 83240000 ns | +66500000 ns (+397.3%) | 16410000 ns | -330000 ns (-2.0%) |
| SM Throughput (%) | 40.72 % | 21.01 % | -19.71 pp | 41.65 % | +0.93 pp | 26.11 % | -14.61 pp | 41.32 % | +0.60 pp |
| Memory Throughput (%) | 59.02 % | 38.50 % | -20.52 pp | 60.13 % | +1.11 pp | 53.63 % | -5.39 pp | 59.65 % | +0.63 pp |
| DRAM Throughput (%) | 1.49 % | 37.87 % | +36.38 pp | 1.55 % | +0.06 pp | 40.16 % | +38.67 pp | 1.55 % | +0.06 pp |
| Achieved Occupancy (%) | 12.34 % | 24.57 % | +12.23 pp | 12.37 % | +0.03 pp | 18.43 % | +6.09 pp | 12.44 % | +0.10 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 128.00 register/thread | -127 | 255.00 register/thread | +0 |
| Waves/SM | 11.64 | 5.82 | -5.82 | 11.64 | +0.00 | 7.76 | -3.88 | 11.64 | +0.00 |
| No Eligible (%) | 58.23 % | 83.64 % | +25.41 pp | 57.75 % | -0.48 pp | 85.52 % | +27.29 pp | 57.10 % | -1.13 pp |
| IPC Active | 1.67 inst/cycle | 0.66 inst/cycle | -1.01 | 1.71 inst/cycle | +0.04 | 0.58 inst/cycle | -1.09 | 1.71 inst/cycle | +0.04 |


### Key Learnings

#### ❌ Plan B: `tma_loads_via_block_ptr` (0.280×)

Replacing scalar pointer arithmetic with tl.make_block_ptr + tl.advance for TMA-style loads produced only a marginal 4% wall-time improvement (62.37ms → 59.86ms) while the overall speedup relative to a faster baseline was 0.28x, making it a net regression. The key profiling metrics barely moved: scheduler No Eligible dropped only from 84.82% to 83.64%, eligible warps per scheduler from 0.19 to 0.21, and warp cycles per issued instruction from 25.75 to 23.96 — all bottlenecks remained. Critically, executed instructions actually increased by 3% (7.36B → 7.58B) and the CPIStall and UncoalescedSharedAccess rules persisted at nearly identical severity, indicating that the block_ptr approach did not eliminate the underlying shared memory bank conflicts (5.3→5.4-way) or the local load/store uncoalesced access pattern (still 1/32 bytes/sector). When a kernel is already register-spill-limited (stack size 2208-2304 bytes, 128 regs/thread at occupancy ceiling), switching to block_ptr loads does not resolve the dominant L1TEX stall source because the stalls originate from spill traffic, not from the global load addressing pattern.

> **Applicable when**: tl.make_block_ptr TMA loads are beneficial when: (1) the dominant stall is global memory latency from uncoalesced scalar pointer loads (not spill traffic), (2) registers_per_thread is well below the hardware maximum (e.g., <128), (3) stack size is small (no spilling), and (4) num_stages >= 3 is available to pipeline the async loads.

> **Avoid when**: Do NOT apply tl.make_block_ptr TMA loads when the kernel is already register-spill-limited: stack size > 1024 bytes, registers_per_thread at occupancy ceiling (128 with Block Limit Registers=4), and CPIStall is driven by local load/store spill traffic rather than global memory access pattern. In this regime, the block_ptr change does not reduce instruction count or stall cycles — it may even increase instruction count — and the dominant bottleneck (spill-induced L1TEX stalls) remains unchanged. The UncoalescedSharedAccess and SharedMemoryConflicts rules also persist, indicating the real bottleneck is shared memory bank conflicts from the matmul tile layout, not the global load addressing.

---

## Cumulative Progress

```
Baseline:  16740000 ns  (1.000×)
Iter 1     16340000 ns  (1.024×)  — tf32_tensor_cores

Final best: 16340000 ns  (1.0245×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4,  num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=8,  num_stages=4, maxnreg=128),
+        triton.Config({}, num_warps=16, num_stages=2, maxnreg=128),
+        triton.Config({}, num_warps=16, num_stages=3, maxnreg=128),
+        triton.Config({}, num_warps=4,  num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=8,  num_stages=2, maxnreg=96),
+        triton.Config({}, num_warps=4,  num_stages=2, maxnreg=64),
+        triton.Config({}, num_warps=8,  num_stages=2, maxnreg=64),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_add_addmm_mul_pow_t_tanh_view_13(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -87,8 +105,6 @@

     xindex = idx_n + 768*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
