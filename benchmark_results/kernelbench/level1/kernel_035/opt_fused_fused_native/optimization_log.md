# Kernel Optimization Log

## Summary

- **Kernel**: `group_norm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7120000 ns — **Best latency**: 7150000 ns (**0.9958×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `group_norm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_035/synthesis/kernel_0_fused_fused_native.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (896,1,1), Waves/SM: 0.85
- **Registers/thread**: 32
- **Duration**: 7120000 ns
- **SM Throughput**: 37.53 % | **Memory**: 64.42 % | **DRAM**: 64.42 %
- **Achieved Occupancy**: 79.05 %
- **IPC Active**: 1.50 inst/cycle | **No Eligible**: 62.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | 0.985× | 0.990× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints | 1.006× | 1.004× | ✓ |  |
| C | memory_coalescing_channel_outer_spatial_inner_loop_reorder | 1.221× | 0.996× | ✓ | **✓ best** |
| D | preload_weight_bias_into_registers_eliminate_redundant_global_loads | 1.008× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | preload_weight_bi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7120000 ns | 7190000 ns | +70000 ns (+1.0%) | 7090000 ns | -30000 ns (-0.4%) | 7150000 ns | +30000 ns (+0.4%) | 7070000 ns | -50000 ns (-0.7%) |
| SM Throughput (%) | 37.53 % | 37.77 % | +0.24 pp | 39.82 % | +2.29 pp | 35.43 % | -2.10 pp | 35.72 % | -1.81 pp |
| Memory Throughput (%) | 64.42 % | 63.73 % | -0.69 pp | 64.64 % | +0.22 pp | 64.16 % | -0.26 pp | 64.83 % | +0.41 pp |
| DRAM Throughput (%) | 64.42 % | 63.73 % | -0.69 pp | 64.64 % | +0.22 pp | 64.16 % | -0.26 pp | 64.83 % | +0.41 pp |
| Achieved Occupancy (%) | 79.05 % | 79.15 % | +0.10 pp | 79.77 % | +0.72 pp | 79.36 % | +0.31 pp | 79.57 % | +0.52 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 31.00 register/thread | -1 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 |
| No Eligible (%) | 62.52 % | 61.29 % | -1.23 pp | 61.32 % | -1.20 pp | 64.07 % | +1.55 pp | 64.11 % | +1.59 pp |
| IPC Active | 1.50 inst/cycle | 1.55 inst/cycle | +0.05 | 1.55 inst/cycle | +0.05 | 1.44 inst/cycle | -0.06 | 1.43 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | single_pass_fused_groupnorm_eliminate_second_dram_read | 1.018× | 1.017× | ✓ |  |
| B | reduce_cta_barrier_stalls_via_restructured_welford_and_pass2_gather | 1.134× | 0.997× | ✓ |  |
| C | single_pass_fused_groupnorm_num_stages_pipelining | 1.219× | 0.999× | ✓ |  |
| D | single_pass_smem_reuse_eliminate_second_dram_read | 0.988× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | single_pass_fused... | Δ | reduce_cta_barrie... | Δ | single_pass_fused... | Δ | single_pass_smem_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7120000 ns | 7000000 ns | -120000 ns (-1.7%) | 7140000 ns | +20000 ns (+0.3%) | 7130000 ns | +10000 ns (+0.1%) | 7130000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 37.53 % | 36.02 % | -1.51 pp | 37.82 % | +0.29 pp | 35.38 % | -2.15 pp | 39.76 % | +2.23 pp |
| Memory Throughput (%) | 64.42 % | 65.51 % | +1.09 pp | 64.20 % | -0.22 pp | 64.28 % | -0.14 pp | 64.30 % | -0.12 pp |
| DRAM Throughput (%) | 64.42 % | 65.51 % | +1.09 pp | 64.20 % | -0.22 pp | 64.28 % | -0.14 pp | 64.30 % | -0.12 pp |
| Achieved Occupancy (%) | 79.05 % | 79.55 % | +0.50 pp | 79.49 % | +0.44 pp | 79.36 % | +0.31 pp | 79.54 % | +0.49 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 31.00 register/thread | -1 |
| Waves/SM | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 | 0.85 | +0.00 |
| No Eligible (%) | 62.52 % | 63.76 % | +1.24 pp | 61.19 % | -1.33 pp | 64.16 % | +1.64 pp | 61.08 % | -1.44 pp |
| IPC Active | 1.50 inst/cycle | 1.45 inst/cycle | -0.05 | 1.55 inst/cycle | +0.05 | 1.43 inst/cycle | -0.07 | 1.55 inst/cycle | +0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eliminate_tl_gather_barrier_stalls_with_direct_channel_load | 0.985× | 0.994× | ✓ |  |
| D | preload_weight_bias_smem_eliminate_per_tile_strided_loads | 1.120× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_gath... | Δ | preload_weight_bi... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7120000 ns | 7160000 ns | +40000 ns (+0.6%) | 7160000 ns | +40000 ns (+0.6%) |
| SM Throughput (%) | 37.53 % | 37.71 % | +0.18 pp | 39.77 % | +2.24 pp |
| Memory Throughput (%) | 64.42 % | 64.00 % | -0.42 pp | 64.03 % | -0.39 pp |
| DRAM Throughput (%) | 64.42 % | 64.00 % | -0.42 pp | 64.03 % | -0.39 pp |
| Achieved Occupancy (%) | 79.05 % | 79.18 % | +0.13 pp | 79.57 % | +0.52 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 31.00 register/thread | -1 |
| Waves/SM | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 |
| No Eligible (%) | 62.52 % | 61.28 % | -1.24 pp | 61.19 % | -1.33 pp |
| IPC Active | 1.50 inst/cycle | 1.54 inst/cycle | +0.04 | 1.55 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  7120000 ns  (1.000×)

Final best: 7150000 ns  (0.9958×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 
 @triton.autotune(
@@ -45,50 +46,54 @@

     base_offset = batch_idx * C * HW + group_idx * C_per_group * HW
 
     # Pass 1: Compute mean and variance using Welford's online algorithm
+    # Restructured to channel-outer / spatial-inner for better memory access
     mean = tl.zeros((), dtype=tl.float32)
     m2 = tl.zeros((), dtype=tl.float32)
     count = tl.zeros((), dtype=tl.float32)
 
-    for start in tl.range(0, group_size, BLOCK_R):
-        offsets = start + tl.arange(0, BLOCK_R)
-        mask = offsets < group_size
-        x = tl.load(X_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
+    for c in tl.range(0, C_per_group):
+        c_offset = c * HW
+        for s_start in tl.range(0, HW, BLOCK_R):
+            offsets = s_start + tl.arange(0, BLOCK_R)
+            mask = offsets < HW
+            x = tl.load(X_ptr + base_offset + c_offset + offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
 
-        # Welford update
-        block_count = tl.sum(mask.to(tl.float32), axis=0)
-        block_mean = tl.sum(tl.where(mask, x, tl.zeros_like(x)), axis=0) / tl.maximum(block_count, 1.0)
-        block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), tl.zeros_like(x)), axis=0)
+            # Welford update
+            block_count = tl.sum(mask.to(tl.float32), axis=0)
+            block_mean = tl.sum(tl.where(mask, x, tl.zeros_like(x)), axis=0) / tl.maximum(block_count, 1.0)
+            block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), tl.zeros_like(x)), axis=0)
 
-        # Merge with running stats
-        new_count = count + block_count
-        delta = block_mean - mean
-        mean = tl.where(new_count > 0, mean + delta * block_count / new_count, mean)
-        m2 = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(new_count, 1.0)
-        count = new_count
+            # Merge with running stats
+            new_count = count + block_count
+            delta = block_mean - mean
+            mean = tl.where(new_count > 0, mean + delta * block_count / new_count, mean)
+            m2 = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(new_count, 1.0)
+            count = new_count
 
     variance = m2 / tl.maximum(count, 1.0)
     inv_std = libdevice.rsqrt(variance + eps)
 
     # Pass 2: Normalize and apply weight/bias
-    for start in tl.range(0, group_size, BLOCK_R):
-        offsets = start + tl.arange(0, BLOCK_R)
-        mask = offsets < group_size
+    # Channel-outer / spatial-inner: w_val and b_val are scalar per channel iteration
+    # This eliminates ~2048 redundant W/B global loads (one per BLOCK_R iteration)
+    for c in tl.range(0, C_per_group):
+        channel_idx = group_idx * C_per_group + c
+        # Load w and b once per channel (scalar), hoisted out of spatial loop
+        # channel_idx is always valid: mask=True (scalar load, always in-bounds)
+        w_val = tl.load(W_ptr + channel_idx, mask=True, other=1.0).to(tl.float32)
+        b_val = tl.load(B_ptr + channel_idx, mask=True, other=0.0).to(tl.float32)
 
-        x = tl.load(X_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
+        c_offset = c * HW
+        for s_start in tl.range(0, HW, BLOCK_R):
+            offsets = s_start + tl.arange(0, BLOCK_R)
+            mask = offsets < HW
 
-        # Compute channel index for weight/bias
-        # offsets maps to: channel_in_group = offsets // HW, spatial = offsets % HW
-        channel_in_group = offsets // HW
-        channel_idx = group_idx * C_per_group + channel_in_group
+            x = tl.load(X_ptr + base_offset + c_offset + offsets, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
 
-        w = tl.load(W_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
-        b = tl.load(B_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
+            x_norm = (x - mean) * inv_std
+            y = x_norm * w_val + b_val
 
-        x_norm = (x - mean) * inv_std
-        y = x_norm * w + b
-
-        tl.store(Y_ptr + base_offset + offsets, y.to(tl.float16 if Y_ptr.dtype.element_ty == tl.float16 else tl.float32), mask=mask)
-
+            tl.store(Y_ptr + base_offset + c_offset + offsets, y.to(tl.float16 if Y_ptr.dtype.element_ty == tl.float16 else tl.float32), mask=mask, eviction_policy='evict_first')
 
 def kernel_function(weight, bias, x):
     N, C, H, W = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
