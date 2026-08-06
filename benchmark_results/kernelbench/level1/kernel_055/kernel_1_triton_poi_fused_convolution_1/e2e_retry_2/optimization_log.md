# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285950 ns — **Best latency**: 233730 ns (**1.2234×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`shared_memory_tiled_transpose_for_coalesced_access`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_055/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285950 ns
- **SM Throughput**: 19.99 % | **Memory**: 78.83 % | **DRAM**: 72.17 %
- **Achieved Occupancy**: 94.04 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_tiled_transpose_for_coalesced_access | 1.620× | 1.223× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285950 ns | 233730 ns | -52220 ns (-18.3%) | 286140 ns | +190 ns (+0.1%) |
| SM Throughput (%) | 19.99 % | 43.87 % | +23.88 pp | 19.80 % | -0.19 pp |
| Memory Throughput (%) | 78.83 % | 53.06 % | -25.77 pp | 78.73 % | -0.10 pp |
| DRAM Throughput (%) | 72.17 % | 21.59 % | -50.58 pp | 72.12 % | -0.05 pp |
| Achieved Occupancy (%) | 94.04 % | 94.11 % | +0.07 pp | 94.00 % | -0.04 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.20 % | 61.60 % | -18.60 pp | 79.58 % | -0.62 pp |
| IPC Active | 0.79 inst/cycle | 1.51 inst/cycle | +0.72 | 0.82 inst/cycle | +0.03 |

---

## Cumulative Progress

```
Baseline:  285950 ns  (1.000×)
Iter 1     233730 ns  (1.223×)  — shared_memory_tiled_transpose_for_coalesced_access

Final best: 233730 ns  (1.2234×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=2),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=2),
+    triton.Config({'YBLOCK': 128, 'XBLOCK': 8}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
-def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
+def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
+    # ynumel = 4096 = B*C = 32*128, xnumel = 31500 = H*W = 126*250
+    # YBLOCK = TILE_C (channel/batch tile, stride-1 in source)
+    # XBLOCK = TILE_X (spatial tile, stride-1 in dest)
+    # Source strides: (4032000, 1, 32000, 128) -> channel dim is stride-1
+    # Dest strides:   (4032000, 31500, 250, 1) -> spatial dim is stride-1
+    pid_x  = tl.program_id(0)  # spatial tile
+    pid_bc = tl.program_id(1)  # batch*channel tile
+    x_off  = pid_x  * XBLOCK
+    bc_off = pid_bc * YBLOCK
+    tc = tl.arange(0, YBLOCK)  # channel axis - stride-1 in source
+    tx = tl.arange(0, XBLOCK)  # spatial axis - stride-1 in dest
+    bc_idx = bc_off + tc        # [YBLOCK]
+    x_idx  = x_off  + tx        # [XBLOCK]
+    b_idx  = bc_idx // 128      # batch index
+    c_idx  = bc_idx % 128       # channel index
+    # Source address: c_idx is stride-1, so loading along YBLOCK dim is coalesced
+    src_addr = c_idx[:, None] + 128 * x_idx[None, :] + 4032000 * b_idx[:, None]  # [YBLOCK, XBLOCK]
+    bc_mask = bc_idx < ynumel   # [YBLOCK]
+    x_mask  = x_idx  < xnumel  # [XBLOCK]
+    mask2d  = bc_mask[:, None] & x_mask[None, :]  # [YBLOCK, XBLOCK]
+    tile = tl.load(in_ptr0 + src_addr, mask=mask2d, other=0.0)  # [YBLOCK, XBLOCK]
+    tile_T = tl.trans(tile)  # [XBLOCK, YBLOCK] - in-register transpose
+    # Dest address: x_idx is stride-1, so storing along XBLOCK dim is coalesced
+    dst_addr = x_idx[:, None] + 31500 * bc_idx[None, :]  # [XBLOCK, YBLOCK]
+    tl.store(out_ptr0 + dst_addr, tile_T, mask=tl.trans(mask2d))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +49,7 @@

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
