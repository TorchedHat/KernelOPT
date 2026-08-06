# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285920 ns — **Best latency**: 232380 ns (**1.2304×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`coalesced_transpose_via_shared_memory_tiling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285920 ns
- **SM Throughput**: 19.56 % | **Memory**: 78.77 % | **DRAM**: 72.16 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesced_transpose_via_shared_memory_tiling | 1.636× | 1.230× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_transpo... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285920 ns | 232380 ns | -53540 ns (-18.7%) |
| SM Throughput (%) | 19.56 % | 44.49 % | +24.93 pp |
| Memory Throughput (%) | 78.77 % | 53.21 % | -25.56 pp |
| DRAM Throughput (%) | 72.16 % | 21.59 % | -50.57 pp |
| Achieved Occupancy (%) | 94.05 % | 94.07 % | +0.02 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.39 % | 61.97 % | -18.42 pp |
| IPC Active | 0.78 inst/cycle | 1.53 inst/cycle | +0.75 |

---

## Cumulative Progress

```
Baseline:  285920 ns  (1.000×)
Iter 1     232380 ns  (1.230×)  — coalesced_transpose_via_shared_memory_tiling

Final best: 232380 ns  (1.2304×)
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
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+], key=['xnumel', 'ynumel'])
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    pid_x = tl.program_id(0)  # tile along x2 (output fast dim)
+    pid_y = tl.program_id(1)  # tile along y3
+    x_offs = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
+    y_offs = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
+    x_mask = x_offs < xnumel
+    y_mask = y_offs < ynumel
+    y0 = y_offs % 128   # fast dim in input (stride=1)
+    y1 = y_offs // 128  # batch dim in input (stride=4032000)
+    # Load [YBLOCK, XBLOCK]: coalesced along y0 (stride=1), strided along x2 (stride=128)
+    load_idx = y0[:, None] + 128 * x_offs[None, :] + 4032000 * y1[:, None]
+    mask2d = y_mask[:, None] & x_mask[None, :]
+    tile = tl.load(in_ptr0 + load_idx, mask=mask2d, other=0.0)
+    # Transpose in SMEM: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
+    tile_T = tl.trans(tile)
+    # Store [XBLOCK, YBLOCK]: coalesced along x2 (stride=1 in output)
+    store_idx = x_offs[:, None] + 31500 * y_offs[None, :]
+    store_mask = x_mask[:, None] & y_mask[None, :]
+    tl.store(out_ptr0 + store_idx, tile_T, mask=store_mask)
 
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
