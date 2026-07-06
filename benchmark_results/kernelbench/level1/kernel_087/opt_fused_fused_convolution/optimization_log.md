# Kernel Optimization Log

## Summary

- **Kernel**: `pointwise_conv2d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 15160000 ns — **Best latency**: 9970000 ns (**1.5206×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `pointwise_conv2d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_087/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131072,1,1), Waves/SM: 496.48
- **Registers/thread**: 255
- **Duration**: 15160000 ns
- **SM Throughput**: 45.56 % | **Memory**: 64.48 % | **DRAM**: 17.43 %
- **Achieved Occupancy**: 12.41 %
- **IPC Active**: 1.82 inst/cycle | **No Eligible**: 54.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.515× | 1.521× | ✓ | **✓ best** |
| C | eviction_policies_streaming_x_reuse_w | 1.001× | 0.998× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tiles_and_more_stages | 0.824× | 0.283× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 15160000 ns | 9970000 ns | -5190000 ns (-34.2%) | 15190000 ns | +30000 ns (+0.2%) | 53490000 ns | +38330000 ns (+252.8%) |
| SM Throughput (%) | 45.56 % | 16.55 % | -29.01 pp | 45.62 % | +0.06 pp | 19.06 % | -26.50 pp |
| Memory Throughput (%) | 64.48 % | 83.49 % | +19.01 pp | 64.53 % | +0.05 pp | 95.42 % | +30.94 pp |
| DRAM Throughput (%) | 17.43 % | 26.26 % | +8.83 pp | 17.25 % | -0.18 pp | 4.89 % | -12.54 pp |
| Achieved Occupancy (%) | 12.41 % | 12.39 % | -0.02 pp | 12.40 % | -0.01 pp | 18.58 % | +6.17 pp |
| Registers/Thread | 255.00 register/thread | 214.00 register/thread | -41 | 255.00 register/thread | +0 | 86.00 register/thread | -169 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 | 2647.92 | +2151.44 |
| No Eligible (%) | 54.32 % | 83.43 % | +29.11 pp | 54.27 % | -0.05 pp | 80.93 % | +26.61 pp |
| IPC Active | 1.82 inst/cycle | 0.66 inst/cycle | -1.16 | 1.83 inst/cycle | +0.01 | 0.76 inst/cycle | -1.06 |

---

## Iteration 2

**Best so far**: 1.5206× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_uncoalesced_x_access_via_host_side_nhwc_pretranspose | 1.509× | 1.519× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_uncoalesced_x... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 15160000 ns | 9980000 ns | -5180000 ns (-34.2%) |
| SM Throughput (%) | 45.56 % | 16.54 % | -29.02 pp |
| Memory Throughput (%) | 64.48 % | 83.46 % | +18.98 pp |
| DRAM Throughput (%) | 17.43 % | 26.23 % | +8.80 pp |
| Achieved Occupancy (%) | 12.41 % | 12.39 % | -0.02 pp |
| Registers/Thread | 255.00 register/thread | 214.00 register/thread | -41 |
| Waves/SM | 496.48 | 496.48 | +0.00 |
| No Eligible (%) | 54.32 % | 83.43 % | +29.11 pp |
| IPC Active | 1.82 inst/cycle | 0.66 inst/cycle | -1.16 |


### Key Learnings

#### ⚠️ Plan A: `fix_uncoalesced_x_access_via_host_side_nhwc_pretranspose` (1.519×)

Performing a host-side x.permute(0,2,3,1).contiguous() to convert input from NCHW to NHWC layout and updating the hw_idx stride in the kernel's x_ptrs line was expected to eliminate 73% excessive shared wavefronts and 43% excessive global sectors. However, every NCU metric after the change is statistically identical to before (L1/TEX throughput 83.58%→83.56%, uncoalesced wavefronts unchanged at 73%, duration 9.97ms→9.98ms), indicating the layout change had no measurable effect on the kernel's memory access pattern. The reported 1.519x speedup is inconsistent with the profiling data and likely reflects measurement noise or a different code path being timed. The root cause of uncoalesced access in this kernel is not resolved by this approach.

> **Applicable when**: Never produced measurable improvement in this configuration. Do not apply when NCU metrics show no change after the layout transformation — the kernel's uncoalesced access pattern persists regardless of host-side pretranspose.

> **Avoid when**: When NCU UncoalescedSharedAccess and UncoalescedGlobalAccess metrics remain unchanged after applying the NHWC pretranspose, the bottleneck is not in the x-tensor access pattern alone. The strategy is ineffective when the kernel's shared memory access pattern (not just global memory layout) is the dominant source of uncoalesced wavefronts, as shared memory bank conflicts are independent of the global memory layout of the input tensor.

---

## Iteration 3

**Best so far**: 1.5206× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_output_store_via_nhwc_output_layout | 1.495× | 1.519× | ✓ |  |
| C | fix_uncoalesced_output_store_via_nhwc_output_layout | 1.505× | 1.519× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_output_s... | Δ | fix_uncoalesced_o... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 15160000 ns | 9980000 ns | -5180000 ns (-34.2%) | 9980000 ns | -5180000 ns (-34.2%) |
| SM Throughput (%) | 45.56 % | 16.03 % | -29.53 pp | 16.03 % | -29.53 pp |
| Memory Throughput (%) | 64.48 % | 83.22 % | +18.74 pp | 83.22 % | +18.74 pp |
| DRAM Throughput (%) | 17.43 % | 26.21 % | +8.78 pp | 26.22 % | +8.79 pp |
| Achieved Occupancy (%) | 12.41 % | 12.37 % | -0.04 pp | 12.37 % | -0.04 pp |
| Registers/Thread | 255.00 register/thread | 214.00 register/thread | -41 | 214.00 register/thread | -41 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 54.32 % | 83.95 % | +29.63 pp | 83.95 % | +29.63 pp |
| IPC Active | 1.82 inst/cycle | 0.64 inst/cycle | -1.18 | 0.64 inst/cycle | -1.18 |


### Key Learnings

#### ⚠️ Plan A: `coalesce_output_store_via_nhwc_output_layout` (1.519×)

Allocating the output tensor in NHWC layout (Co innermost, stride_oc=1) and updating the kernel's store pointer arithmetic was expected to eliminate the 43% excessive global sectors flagged by UncoalescedGlobalAccess. However, every NCU metric after the change is statistically identical to before: L1/TEX throughput 83.56%→83.3%, DRAM throughput 26.23%→26.21%, duration 9.98ms→9.98ms, and UncoalescedGlobalAccess excessive sectors remain at 43%. The reported 1.519x speedup is inconsistent with the profiling data and likely reflects measurement noise or a different code path being timed. The root cause of uncoalesced global access in this kernel is not resolved by changing the output tensor layout alone — the 73% UncoalescedSharedAccess from tl.dot internal shared memory and the structural access pattern of the GEMM tile remain the dominant bottlenecks.

> **Applicable when**: Never produced measurable improvement in this configuration. Do not apply when NCU UncoalescedGlobalAccess metrics remain unchanged after the output layout transformation — the kernel's uncoalesced access pattern persists regardless of output tensor layout change.

> **Avoid when**: When NCU UncoalescedGlobalAccess and UncoalescedSharedAccess metrics remain unchanged after applying the NHWC output layout, the bottleneck is not in the output store path alone. The strategy is ineffective when the kernel's shared memory access pattern (tl.dot tile layout, 73% excessive wavefronts) is the dominant source of uncoalesced traffic, as shared memory bank conflicts are independent of the global memory layout of the output tensor. Also avoid when the permute().contiguous() overhead on the output adds non-trivial cost.

#### ⚠️ Plan C: `fix_uncoalesced_output_store_via_nhwc_output_layout` (1.519×)

Switching the output tensor from NCHW to NHWC layout (stride_oc=1) and transposing the weight tensor to (Ci, Co) was expected to eliminate the 43% excessive global sectors and 73% excessive shared wavefronts flagged by NCU. However, every key NCU metric after the change is statistically identical to before: L1/TEX throughput 83.56%→83.36%, uncoalesced global sectors unchanged at 43%, uncoalesced shared wavefronts unchanged at 73%, duration 9.98ms→9.98ms. The reported 1.519x speedup is inconsistent with the profiling data and likely reflects wall-clock measurement noise rather than a genuine kernel improvement. The dominant bottleneck (UncoalescedSharedAccess at 73% excessive wavefronts) originates from shared memory bank conflicts inside tl.dot tile operations, which are independent of the global output memory layout.

> **Applicable when**: Never produced measurable improvement in this configuration. NCU metrics are unchanged after applying NHWC output layout transformation, indicating the bottleneck is not in the output store path.

> **Avoid when**: When NCU UncoalescedSharedAccess (73% excessive wavefronts) and UncoalescedGlobalAccess (43% excessive sectors) metrics remain unchanged after applying the NHWC output layout transformation, the bottleneck is not in the output store. The strategy is ineffective when the dominant source of uncoalesced wavefronts is shared memory bank conflicts inside tl.dot tile operations (which are independent of global output memory layout). Do not apply when the kernel's shared memory access pattern is the primary bottleneck.

---

## Cumulative Progress

```
Baseline:  15160000 ns  (1.000×)
Iter 1     9970000 ns  (1.521×)  — tf32_tensor_cores

Final best: 9970000 ns  (1.5206×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -13,6 +14,8 @@

         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -30,7 +33,7 @@

     # M = N_batch * H * W (output spatial positions)
     # N = out_channels (Co)
     # K = in_channels (Ci)
-    # x layout: (N_batch, Ci, H, W) with strides stride_xn, stride_xc, stride_xh, stride_xw
+    # x layout: (N_batch, H, W, Ci) [NHWC] with strides stride_xn, stride_xc, stride_xh, stride_xw
     # w layout: (Co, Ci) with strides stride_wco, stride_wci
     # out layout: (N_batch, Co, H, W) with strides stride_on, stride_oc, stride_oh, stride_ow
 
@@ -53,13 +56,6 @@

     # offs_m = batch_idx * HW + hw_idx
     batch_idx = offs_m // HW   # [BLOCK_M]
     hw_idx = offs_m % HW       # [BLOCK_M]
-    h_idx = hw_idx // tl.cdiv(HW, HW)  # placeholder, compute properly
-    # Actually HW = H*W, hw_idx = h*W + w
-    # We need h and w for the x pointer
-    # x_ptr[batch, ci, h, w] = x_ptr + batch*stride_xn + ci*stride_xc + h*stride_xh + w*stride_xw
-    # Since stride_xh = W and stride_xw = 1 (NCHW contiguous):
-    # x_ptr[batch, ci, hw] = x_ptr + batch*stride_xn + ci*stride_xc + hw  (since h*W+w = hw_idx)
-    # This works because stride_xh * h + stride_xw * w = W*h + w = hw_idx when stride_xw=1
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
@@ -67,12 +63,13 @@

         k_offs = k * BLOCK_K + offs_k  # [BLOCK_K]
 
         # Load x tile: shape [BLOCK_M, BLOCK_K]
-        # x[batch_idx[m], k_offs[k], hw_idx[m]]
-        # ptr = x_ptr + batch_idx[m]*stride_xn + k_offs[k]*stride_xc + hw_idx[m]
+        # x[batch_idx[m], hw_idx[m], k_offs[k]] in NHWC layout
+        # ptr = x_ptr + batch_idx[m]*stride_xn + k_offs[k]*stride_xc + hw_idx[m]*stride_xw
+        # stride_xc = 1 (Ci innermost in NHWC), stride_xw = Ci
         x_batch = batch_idx[:, None]   # [BLOCK_M, 1]
         x_ci = k_offs[None, :]         # [1, BLOCK_K]
         x_hw = hw_idx[:, None]         # [BLOCK_M, 1]
-        x_ptrs = x_ptr + x_batch.to(tl.int64) * stride_xn + x_ci.to(tl.int64) * stride_xc + x_hw.to(tl.int64)
+        x_ptrs = x_ptr + x_batch.to(tl.int64) * stride_xn + x_ci.to(tl.int64) * stride_xc + x_hw.to(tl.int64) * stride_xw
         x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
         x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
 
@@ -84,7 +81,7 @@

         w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
         w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
 
-        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
+        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)
 
     # Store output: out[batch_idx[m], offs_n[n], hw_idx[m]]
     # out_ptr + batch_idx[m]*stride_on + offs_n[n]*stride_oc + hw_idx[m]
@@ -94,7 +91,6 @@

     out_ptrs = out_ptr + out_batch.to(tl.int64) * stride_on + out_co.to(tl.int64) * stride_oc + out_hw.to(tl.int64)
     out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)
-
 
 def kernel_function(w, x):
     # w: (Co, Ci, 1, 1), x: (N, Ci, H, W)
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
