# Kernel Optimization Log

## Summary

- **Kernel**: `flash_attn_fwd_kernel_split_d` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 50880000 ns — **Best latency**: 50820000 ns (**1.0012×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_tile_sizes_and_add_eviction_policies_with_pipelining`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `flash_attn_fwd_kernel_split_d`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_097/synthesis/kernel_0_aten_scaled_dot_product_attention_defaul.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (32768,1,1), Waves/SM: 248.24
- **Registers/thread**: 157
- **Duration**: 50880000 ns
- **SM Throughput**: 36.35 % | **Memory**: 77.63 % | **DRAM**: 1.72 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.46 inst/cycle | **No Eligible**: 63.57 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | increase_tile_sizes_and_add_eviction_policies_with_pipelining | 1.004× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 50880000 ns | 50820000 ns | -60000 ns (-0.1%) |
| SM Throughput (%) | 36.35 % | 36.32 % | -0.03 pp |
| Memory Throughput (%) | 77.63 % | 77.56 % | -0.07 pp |
| DRAM Throughput (%) | 1.72 % | 1.72 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp |
| Registers/Thread | 157.00 register/thread | 157.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 |
| No Eligible (%) | 63.57 % | 63.61 % | +0.04 pp |
| IPC Active | 1.46 inst/cycle | 1.46 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_num_warps_reduction | 1.002× | 0.998× | ✓ |  |
| C | reduce_register_pressure_via_d_head_chunking | 1.002× | 0.997× | ✓ |  |
| D | reduce_num_warps_to_improve_occupancy_under_register_pressure | 1.003× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 50880000 ns | 50980000 ns | +100000 ns (+0.2%) | 51030000 ns | +150000 ns (+0.3%) | 50930000 ns | +50000 ns (+0.1%) |
| SM Throughput (%) | 36.35 % | 36.30 % | -0.05 pp | 36.11 % | -0.24 pp | 36.31 % | -0.04 pp |
| Memory Throughput (%) | 77.63 % | 77.51 % | -0.12 pp | 77.11 % | -0.52 pp | 77.54 % | -0.09 pp |
| DRAM Throughput (%) | 1.72 % | 1.71 % | -0.01 pp | 1.71 % | -0.01 pp | 1.71 % | -0.01 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 157.00 register/thread | 157.00 register/thread | +0 | 157.00 register/thread | +0 | 157.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 248.24 | +0.00 | 248.24 | +0.00 |
| No Eligible (%) | 63.57 % | 63.62 % | +0.05 pp | 63.71 % | +0.14 pp | 63.58 % | +0.01 pp |
| IPC Active | 1.46 inst/cycle | 1.46 inst/cycle | +0.00 | 1.45 inst/cycle | -0.01 | 1.45 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0012× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | static_launch_parameter_tuning_with_safe_constexpr_alignment | 1.002× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | static_launch_par... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 50880000 ns | 50750000 ns | -130000 ns (-0.3%) |
| SM Throughput (%) | 36.35 % | 36.16 % | -0.19 pp |
| Memory Throughput (%) | 77.63 % | 77.22 % | -0.41 pp |
| DRAM Throughput (%) | 1.72 % | 1.72 % | +0.00 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp |
| Registers/Thread | 157.00 register/thread | 157.00 register/thread | +0 |
| Waves/SM | 248.24 | 248.24 | +0.00 |
| No Eligible (%) | 63.57 % | 63.63 % | +0.06 pp |
| IPC Active | 1.46 inst/cycle | 1.45 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  50880000 ns  (1.000×)
Iter 1     50820000 ns  (1.001×)  — increase_tile_sizes_and_add_eviction_policies_with_pipelining

Final best: 50820000 ns  (1.0012×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -6,6 +6,7 @@

 # Flash Attention 2 implementation in Triton
 # Q: (B, H, N, D), K: (B, H, N, D), V: (B, H, N, D)
 # Output: (B, H, N, D)
+
 
 @triton.jit
 def flash_attn_fwd_kernel(
@@ -25,12 +26,6 @@

     pid_m = tl.program_id(0)  # which block of queries
     pid_bh = tl.program_id(1)  # which (batch, head) pair
 
-    # Decode batch and head from pid_bh
-    # We'll pass B*H as the second grid dimension
-    # pid_bh = b * H + h
-    # We need to know H at compile time or pass it
-    # Instead, we use strides to navigate
-
     # Offsets for this block
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_d = tl.arange(0, BLOCK_D)
@@ -44,12 +39,9 @@

     # Load Q block: [BLOCK_M, BLOCK_D]
     q_ptrs = Q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
     mask_m = offs_m < N_CTX
-    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0).to(tl.float32)
+    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0, eviction_policy='evict_last').to(tl.float32)
 
     # Initialize accumulators
-    # m_i: running max, shape [BLOCK_M]
-    # l_i: running sum of exp, shape [BLOCK_M]
-    # acc: running weighted sum of V, shape [BLOCK_M, BLOCK_D]
     m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
     acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
@@ -64,10 +56,9 @@

 
         # Load K block: [BLOCK_N, BLOCK_D]
         k_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
-        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0).to(tl.float32)
+        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0, eviction_policy='evict_first').to(tl.float32)
 
         # Compute QK^T: [BLOCK_M, BLOCK_N]
-        # q: [BLOCK_M, BLOCK_D], k: [BLOCK_N, BLOCK_D]
         qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
 
         # Apply mask for out-of-bounds keys
@@ -85,7 +76,7 @@

 
         # Load V block: [BLOCK_N, BLOCK_D]
         v_ptrs = V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
-        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0).to(tl.float32)
+        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0, eviction_policy='evict_first').to(tl.float32)
 
         # Update accumulator
         l_ij = tl.sum(p, axis=1)  # [BLOCK_M]
@@ -103,6 +94,7 @@

 
 
 @triton.jit
+
 def flash_attn_fwd_kernel_split_d(
     Q_ptr, K_ptr, V_ptr, Out_ptr,
     stride_qb, stride_qh, stride_qn, stride_qd,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
