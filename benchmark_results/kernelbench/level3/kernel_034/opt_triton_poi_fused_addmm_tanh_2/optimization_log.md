# Kernel Optimization Log

## Summary

- **Kernel**: `rnn_i2h_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 47010 ns — **Best latency**: 45440 ns (**1.0346×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_weight_load_stride_to_eliminate_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `rnn_i2h_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_034/synthesis/kernel_2_triton_poi_fused_addmm_tanh_2.py`
- **Framework**: triton
- **Launch**: Block (64,1,1), Grid (4,1,1), Waves/SM: 0.01
- **Registers/thread**: 114
- **Duration**: 47010 ns
- **SM Throughput**: 0.34 % | **Memory**: 2.08 % | **DRAM**: 0.60 %
- **Achieved Occupancy**: 3.12 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 76.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_fuse_rnn_timestep_loop | 0.995× | 0.999× | ✓ |  |
| C | split_k_parallelism | 1.001× | 1.000× | ✓ | **✓ best** |
| D | increase_tile_sizes_and_grid_parallelism | 0.997× | 0.593× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | split_k_paralleli... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 47010 ns | 47040 ns | +30 ns (+0.1%) | 47010 ns | +0 ns (+0.0%) | 79260 ns | +32250 ns (+68.6%) |
| SM Throughput (%) | 0.34 % | 0.34 % | +0.00 pp | 0.34 % | +0.00 pp | 0.62 % | +0.28 pp |
| Memory Throughput (%) | 2.08 % | 2.07 % | -0.01 pp | 2.07 % | -0.01 pp | 4.58 % | +2.50 pp |
| DRAM Throughput (%) | 0.60 % | 0.60 % | +0.00 pp | 0.60 % | +0.00 pp | 0.35 % | -0.25 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp | 6.25 % | +3.13 pp |
| Registers/Thread | 114.00 register/thread | 114.00 register/thread | +0 | 114.00 register/thread | +0 | 255.00 register/thread | +141 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.06 | +0.05 |
| No Eligible (%) | 76.78 % | 76.66 % | -0.12 pp | 76.74 % | -0.04 pp | 89.44 % | +12.66 pp |
| IPC Active | 0.47 inst/cycle | 0.47 inst/cycle | +0.00 | 0.47 inst/cycle | +0.00 | 0.42 inst/cycle | -0.05 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_weight_load_stride_to_eliminate_shared_memory_bank_conflicts | 0.966× | 1.035× | ✓ | **✓ best** |
| B | reduce_shared_memory_bank_conflicts_via_weight_load_transposition | 0.998× | 1.030× | ✓ |  |
| C | reduce_register_pressure_via_single_dot_and_constrained_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_weight_load_s... | Δ | reduce_shared_mem... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 47010 ns | 45440 ns | -1570 ns (-3.3%) | 45660 ns | -1350 ns (-2.9%) |
| SM Throughput (%) | 0.34 % | 0.34 % | +0.00 pp | 0.34 % | +0.00 pp |
| Memory Throughput (%) | 2.08 % | 2.15 % | +0.07 pp | 2.15 % | +0.07 pp |
| DRAM Throughput (%) | 0.60 % | 0.62 % | +0.02 pp | 0.61 % | +0.01 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp | 3.12 % | +0.00 pp |
| Registers/Thread | 114.00 register/thread | 112.00 register/thread | -2 | 112.00 register/thread | -2 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 76.78 % | 76.55 % | -0.23 pp | 76.53 % | -0.25 pp |
| IPC Active | 0.47 inst/cycle | 0.47 inst/cycle | +0.00 | 0.47 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0346× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_weight_load_layout_fix | 0.930× | 0.997× | ✓ |  |
| C | reduce_register_pressure_via_split_k_loop_in_i2h_kernel | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 47010 ns | 47170 ns | +160 ns (+0.3%) |
| SM Throughput (%) | 0.34 % | 0.34 % | +0.00 pp |
| Memory Throughput (%) | 2.08 % | 2.07 % | -0.01 pp |
| DRAM Throughput (%) | 0.60 % | 0.59 % | -0.01 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp |
| Registers/Thread | 114.00 register/thread | 114.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 76.78 % | 76.69 % | -0.09 pp |
| IPC Active | 0.47 inst/cycle | 0.47 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0346× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fuse_i2h_and_h2o_into_single_rnn_step_kernel | 1.019× | 0.999× | ✓ |  |
| E | add_triton_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_i2h_and_h2o_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 47010 ns | 47040 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 0.34 % | 0.34 % | +0.00 pp |
| Memory Throughput (%) | 2.08 % | 2.08 % | +0.00 pp |
| DRAM Throughput (%) | 0.60 % | 0.60 % | +0.00 pp |
| Achieved Occupancy (%) | 3.12 % | 3.12 % | +0.00 pp |
| Registers/Thread | 114.00 register/thread | 114.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 76.78 % | 76.81 % | +0.03 pp |
| IPC Active | 0.47 inst/cycle | 0.47 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  47010 ns  (1.000×)
Iter 2     45440 ns  (1.035×)  — fix_weight_load_stride_to_eliminate_shared_memory_bank_conflicts

Final best: 45440 ns  (1.0346×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,21 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32,  'BLOCK_K': 128}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
     ],
     key=['M', 'N_i2h', 'K_i2h'],
 )
@@ -40,10 +47,11 @@

     for k in range(0, tl.cdiv(K_i2h, BLOCK_K)):
         offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
 
-        # Load weight tile [BLOCK_K, BLOCK_N] -> weight is [N_i2h, K_i2h], we need [K_i2h, N_i2h]
+        # Load weight tile [BLOCK_N, BLOCK_K] -> weight is [N_i2h, K_i2h]
+        # Load with N as outer dim to avoid shared memory bank conflicts
         # weight[n, k] accessed as weight_ptr + n * stride_w_out + k * stride_w_in
-        w_mask = (offs_k[:, None] < K_i2h) & (offs_n[None, :] < N_i2h)
-        w = tl.load(weight_ptr + offs_k[:, None] * stride_w_in + offs_n[None, :] * stride_w_out,
+        w_mask = (offs_n[:, None] < N_i2h) & (offs_k[None, :] < K_i2h)
+        w = tl.load(weight_ptr + offs_n[:, None] * stride_w_out + offs_k[None, :] * stride_w_in,
                     mask=w_mask, other=0.0)
 
         # Load input tile from cat(x, hidden): [BLOCK_M, BLOCK_K]
@@ -65,7 +73,7 @@

 
         inp = tl.where(offs_k[None, :] < K_x, x_val, h_val)
 
-        acc += tl.dot(inp, w, allow_tf32=False)
+        acc += tl.dot(inp, tl.trans(w), allow_tf32=False)
 
     # Add bias and tanh
     bias_mask = offs_n < N_i2h
@@ -80,16 +88,23 @@

 
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
-        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=2, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32,  'BLOCK_K': 64},  num_warps=2, num_stages=4),
         triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32,  'BLOCK_K': 128}, num_warps=2, num_stages=4),
     ],
     key=['M', 'N_h2o', 'K_h2o'],
 )
 @triton.jit
+
 def rnn_h2o_kernel(
     hidden_ptr, weight_ptr, bias_ptr, out_ptr,
     M, N_h2o, K_h2o,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
