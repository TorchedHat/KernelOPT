# Kernel Optimization Log

## Summary

- **Kernel**: `sigmoid_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 3020000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `sigmoid_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_021/synthesis/kernel_0_triton_poi_fused_sigmoid_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 21
- **Duration**: 3020000 ns
- **SM Throughput**: 34.11 % | **Memory**: 86.60 % | **DRAM**: 86.60 %
- **Achieved Occupancy**: 86.08 %
- **IPC Active**: 1.29 inst/cycle | **No Eligible**: 67.83 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_l2 | 0.976× | 0.993× | ✓ |  |
| B | vectorized_loads_with_contiguity_hints | 1.001× | 1.000× | ✓ |  |
| C | increase_memory_pipeline_depth_with_num_stages_and_wider_blocks | 1.001× | 1.000× | ✓ | **✓ best** |
| D | increase_block_size_and_add_software_pipelining | 1.001× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_memory_p... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3020000 ns | +0 ns (+0.0%) | 3020000 ns | +0 ns (+0.0%) | 3080000 ns | +60000 ns (+2.0%) |
| SM Throughput (%) | 34.11 % | 35.18 % | +1.07 pp | 34.15 % | +0.04 pp | 33.75 % | -0.36 pp | 33.38 % | -0.73 pp |
| Memory Throughput (%) | 86.60 % | 86.00 % | -0.60 pp | 86.58 % | -0.02 pp | 86.54 % | -0.06 pp | 84.86 % | -1.74 pp |
| DRAM Throughput (%) | 86.60 % | 86.00 % | -0.60 pp | 86.58 % | -0.02 pp | 86.54 % | -0.06 pp | 84.86 % | -1.74 pp |
| Achieved Occupancy (%) | 86.08 % | 86.11 % | +0.03 pp | 86.07 % | -0.01 pp | 85.98 % | -0.10 pp | 39.48 % | -46.60 pp |
| Registers/Thread | 21.00 register/thread | 20.00 register/thread | -1 | 21.00 register/thread | +0 | 21.00 register/thread | +0 | 52.00 register/thread | +31 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 67.83 % | 65.15 % | -2.68 pp | 67.80 % | -0.03 pp | 67.83 % | +0.00 pp | 69.84 % | +2.01 pp |
| IPC Active | 1.29 inst/cycle | 1.40 inst/cycle | +0.11 | 1.29 inst/cycle | +0.00 | 1.29 inst/cycle | +0.00 | 1.19 inst/cycle | -0.10 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_fp32_sigmoid_via_builtin_and_l2_compression | 1.001× | 1.000× | ✓ |  |
| B | force_large_blocks_with_l2_compression_hints | 0.968× | 0.977× | ✓ |  |
| C | fused_sigmoid_instruction_and_larger_block_autotune | 0.976× | 1.000× | ✓ |  |
| D | use_tl_sigmoid_intrinsic_to_reduce_register_pressure_and_enable_fma_fusion | 1.001× | 0.981× | ✓ |  |
| E | add_triton_autotune | 1.001× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fused_fp32_sigmoi... | Δ | force_large_block... | Δ | fused_sigmoid_ins... | Δ | use_tl_sigmoid_in... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3020000 ns | +0 ns (+0.0%) | 3090000 ns | +70000 ns (+2.3%) | 3020000 ns | +0 ns (+0.0%) | 3080000 ns | +60000 ns (+2.0%) | 3020000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 34.11 % | 33.89 % | -0.22 pp | 34.07 % | -0.04 pp | 39.26 % | +5.15 pp | 33.38 % | -0.73 pp | 33.75 % | -0.36 pp |
| Memory Throughput (%) | 86.60 % | 86.64 % | +0.04 pp | 84.62 % | -1.98 pp | 86.62 % | +0.02 pp | 84.85 % | -1.75 pp | 86.62 % | +0.02 pp |
| DRAM Throughput (%) | 86.60 % | 86.64 % | +0.04 pp | 84.62 % | -1.98 pp | 86.62 % | +0.02 pp | 84.85 % | -1.75 pp | 86.62 % | +0.02 pp |
| Achieved Occupancy (%) | 86.08 % | 86.00 % | -0.08 pp | 85.99 % | -0.09 pp | 83.05 % | -3.03 pp | 39.43 % | -46.65 pp | 85.95 % | -0.13 pp |
| Registers/Thread | 21.00 register/thread | 21.00 register/thread | +0 | 20.00 register/thread | -1 | 16.00 register/thread | -5 | 52.00 register/thread | +31 | 21.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 1489.45 | +744.72 | 372.36 | -372.37 | 744.73 | +0.00 |
| No Eligible (%) | 67.83 % | 67.47 % | -0.36 pp | 65.82 % | -2.01 pp | 61.18 % | -6.65 pp | 69.84 % | +2.01 pp | 67.81 % | -0.02 pp |
| IPC Active | 1.29 inst/cycle | 1.29 inst/cycle | +0.00 | 1.37 inst/cycle | +0.08 | 1.55 inst/cycle | +0.26 | 1.19 inst/cycle | -0.10 | 1.30 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_mlp_via_larger_block_sizes_in_autotune | 1.001× | 1.000× | ✓ |  |
| B | increase_num_stages_for_l1tex_latency_hiding_with_larger_blocks | 0.976× | 1.000× | ✓ |  |
| C | reduce_register_pressure_via_fp16_intermediate_sigmoid | 1.001× | 0.971× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_mlp_via_... | Δ | increase_num_stag... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3020000 ns | +0 ns (+0.0%) | 3020000 ns | +0 ns (+0.0%) | 3110000 ns | +90000 ns (+3.0%) |
| SM Throughput (%) | 34.11 % | 33.73 % | -0.38 pp | 39.28 % | +5.17 pp | 33.29 % | -0.82 pp |
| Memory Throughput (%) | 86.60 % | 86.63 % | +0.03 pp | 86.62 % | +0.02 pp | 84.13 % | -2.47 pp |
| DRAM Throughput (%) | 86.60 % | 86.63 % | +0.03 pp | 86.62 % | +0.02 pp | 84.13 % | -2.47 pp |
| Achieved Occupancy (%) | 86.08 % | 85.94 % | -0.14 pp | 83.02 % | -3.06 pp | 60.68 % | -25.40 pp |
| Registers/Thread | 21.00 register/thread | 21.00 register/thread | +0 | 16.00 register/thread | -5 | 39.00 register/thread | +18 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 1489.45 | +744.72 | 248.24 | -496.49 |
| No Eligible (%) | 67.83 % | 67.67 % | -0.16 pp | 61.16 % | -6.67 pp | 66.37 % | -1.46 pp |
| IPC Active | 1.29 inst/cycle | 1.32 inst/cycle | +0.03 | 1.55 inst/cycle | +0.26 | 1.35 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_loads_evict_last_output_stores | 0.976× | 0.993× | ✓ |  |
| C | remove_bf16_cast_and_use_native_fp32_sigmoid_for_fma_fusion | 1.001× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | remove_bf16_cast_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3080000 ns | +60000 ns (+2.0%) |
| SM Throughput (%) | 34.11 % | 34.80 % | +0.69 pp | 32.99 % | -1.12 pp |
| Memory Throughput (%) | 86.60 % | 85.97 % | -0.63 pp | 84.89 % | -1.71 pp |
| DRAM Throughput (%) | 86.60 % | 85.97 % | -0.63 pp | 84.89 % | -1.71 pp |
| Achieved Occupancy (%) | 86.08 % | 85.93 % | -0.15 pp | 39.62 % | -46.46 pp |
| Registers/Thread | 21.00 register/thread | 20.00 register/thread | -1 | 52.00 register/thread | +31 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 67.83 % | 65.15 % | -2.68 pp | 69.54 % | +1.71 pp |
| IPC Active | 1.29 inst/cycle | 1.39 inst/cycle | +0.10 | 1.19 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)

Final best: 3020000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,32 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        # --- keep all existing configs unchanged ---
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 128},   num_warps=2,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
+        # --- NEW: very large blocks for memory-level parallelism ---
+        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 65536}, num_warps=16, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 65536}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 65536}, num_warps=32, num_stages=3),
     ],
     key=['n_elements'],
 )
@@ -25,14 +43,13 @@

     block_start = pid * BLOCK_SIZE
     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
+    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
     x = tl.load(input_ptr + offsets, mask=mask)
-    # sigmoid(x) = 1 / (1 + exp(-x))
+    # tl.sigmoid() lowers to hardware-fused ex2.approx + rcp on SFU
+    # eliminates separate neg/exp/add/div chain (4 non-fused ops -> 1 fused sequence)
     x_f32 = x.to(tl.float32)
-    neg_x = -x_f32
-    exp_neg_x = tl.exp(neg_x)
-    result = 1.0 / (1.0 + exp_neg_x)
+    result = tl.sigmoid(x_f32)
     tl.store(output_ptr + offsets, result.to(output_ptr.dtype.element_ty), mask=mask)
-
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
