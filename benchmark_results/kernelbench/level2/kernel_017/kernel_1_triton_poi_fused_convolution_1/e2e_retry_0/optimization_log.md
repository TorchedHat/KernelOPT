# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285820 ns — **Best latency**: 143520 ns (**1.9915×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`shared_memory_tiled_transpose_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_017/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285820 ns
- **SM Throughput**: 19.63 % | **Memory**: 78.91 % | **DRAM**: 72.18 %
- **Achieved Occupancy**: 94.07 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_for_coalesced_access | 2.387× | 1.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285820 ns | 143520 ns | -142300 ns (-49.8%) |
| SM Throughput (%) | 19.63 % | 47.63 % | +28.00 pp |
| Memory Throughput (%) | 78.91 % | 41.58 % | -37.33 pp |
| DRAM Throughput (%) | 72.18 % | 34.28 % | -37.90 pp |
| Achieved Occupancy (%) | 94.07 % | 89.97 % | -4.10 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.07 % | 56.63 % | -23.44 pp |
| IPC Active | 0.78 inst/cycle | 1.75 inst/cycle | +0.97 |

---

## Cumulative Progress

```
Baseline:  285820 ns  (1.000×)
Iter 1     143520 ns  (1.992×)  — shared_memory_tiled_transpose_for_coalesced_access

Final best: 143520 ns  (1.9915×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.arange(0, XBLOCK)   # shape [XBLOCK]
+    yindex = yoffset + tl.arange(0, YBLOCK)   # shape [YBLOCK]
     xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    ymask = yindex < ynumel
+    x2 = xindex                    # spatial flat index (stride-1 in output)
+    y3 = yindex                    # full y index
+    y0 = yindex % 128              # channel within batch (stride-1 in input!)
+    y1 = yindex // 128             # batch index
+    # 2D tile: rows=x2 (XBLOCK), cols=y0/y3 (YBLOCK)
+    load_addr = y0[None, :] + 128 * x2[:, None] + 4032000 * y1[None, :]
+    load_mask = xmask[:, None] & ymask[None, :]
+    # Coalesced load: consecutive threads along YBLOCK read consecutive y0 (stride-1)
+    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)
+    # Coalesced store: consecutive threads along XBLOCK write consecutive x2 (stride-1)
+    store_addr = x2[:, None] + 31500 * y3[None, :]
+    tl.store(out_ptr0 + store_addr, tile, mask=load_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
