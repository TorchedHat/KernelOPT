# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_qkv_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3510000 ns — **Best latency**: 3530000 ns (**0.9943×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_qkv_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_050/synthesis/kernel_0_triton_tem_fused_addmm_t_view_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (2304,1,1), Waves/SM: 17.45
- **Registers/thread**: 248
- **Duration**: 3510000 ns
- **SM Throughput**: 37.32 % | **Memory**: 82.62 % | **DRAM**: 1.14 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.54 inst/cycle | **No Eligible**: 61.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_reduce_smem_bank_conflicts_and_register_pressure | 0.980× | 0.989× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_pretranspose_and_eviction_policy | 1.157× | 0.994× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_smaller_attention_tiles | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_reduce_... | Δ | eliminate_shared_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3510000 ns | 3550000 ns | +40000 ns (+1.1%) | 3530000 ns | +20000 ns (+0.6%) | 3510000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.32 % | 36.97 % | -0.35 pp | 37.33 % | +0.01 pp | 37.34 % | +0.02 pp |
| Memory Throughput (%) | 82.62 % | 81.86 % | -0.76 pp | 82.64 % | +0.02 pp | 82.66 % | +0.04 pp |
| DRAM Throughput (%) | 1.14 % | 1.11 % | -0.03 pp | 1.15 % | +0.01 pp | 1.14 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 248.00 register/thread | 254.00 register/thread | +6 | 248.00 register/thread | +0 | 248.00 register/thread | +0 |
| Waves/SM | 17.45 | 17.45 | +0.00 | 17.45 | +0.00 | 17.45 | +0.00 |
| No Eligible (%) | 61.48 % | 61.84 % | +0.36 pp | 61.48 % | +0.00 pp | 61.48 % | +0.00 pp |
| IPC Active | 1.54 inst/cycle | 1.53 inst/cycle | -0.01 | 1.54 inst/cycle | +0.00 | 1.54 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cast_attention_weights_to_fp16_for_tensor_core_mma | 1.104× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cast_attention_we... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3510000 ns | 3520000 ns | +10000 ns (+0.3%) |
| SM Throughput (%) | 37.32 % | 37.32 % | +0.00 pp |
| Memory Throughput (%) | 82.62 % | 82.63 % | +0.01 pp |
| DRAM Throughput (%) | 1.14 % | 1.14 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp |
| Registers/Thread | 248.00 register/thread | 248.00 register/thread | +0 |
| Waves/SM | 17.45 | 17.45 | +0.00 |
| No Eligible (%) | 61.48 % | 61.48 % | +0.00 pp |
| IPC Active | 1.54 inst/cycle | 1.54 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_and_eliminate_redundant_masking | 1.103× | 1.000× | ✓ |  |
| C | eviction_policies_streaming_kv_evict_first | 1.106× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3510000 ns | 3510000 ns | +0 ns (+0.0%) | 3510000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 37.32 % | 37.31 % | -0.01 pp | 37.31 % | -0.01 pp |
| Memory Throughput (%) | 82.62 % | 82.60 % | -0.02 pp | 82.59 % | -0.03 pp |
| DRAM Throughput (%) | 1.14 % | 1.14 % | +0.00 pp | 1.15 % | +0.01 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 248.00 register/thread | 248.00 register/thread | +0 | 248.00 register/thread | +0 |
| Waves/SM | 17.45 | 17.45 | +0.00 | 17.45 | +0.00 |
| No Eligible (%) | 61.48 % | 61.48 % | +0.00 pp | 61.48 % | +0.00 pp |
| IPC Active | 1.54 inst/cycle | 1.54 inst/cycle | +0.00 | 1.54 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  3510000 ns  (1.000×)

Final best: 3530000 ns  (0.9943×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 import math
+
 
 
 @triton.autotune(
@@ -69,14 +70,22 @@

 # We tile over the T dimension for Q and output
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_T': 16, 'BLOCK_S': 16, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_T': 16, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 16, 'BLOCK_D': 64}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_T': 32, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 32, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
         triton.Config({'BLOCK_T': 64, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
+        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 128, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
     ],
     key=['T', 'D'],
 )
 @triton.jit
+
 def attention_relu_kernel(
     Q_ptr, K_ptr, V_ptr, bias_ptr, Out_ptr,
     B, NH, T, D,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
