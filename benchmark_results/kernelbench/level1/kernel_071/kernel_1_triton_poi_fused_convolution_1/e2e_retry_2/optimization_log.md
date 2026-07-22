# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286020 ns — **Best latency**: 234240 ns (**1.2211×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`coalesced_transpose_with_tl_trans_and_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_071/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286020 ns
- **SM Throughput**: 19.58 % | **Memory**: 78.82 % | **DRAM**: 72.15 %
- **Achieved Occupancy**: 94.02 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesced_transpose_with_tl_trans_and_autotune | 1.283× | 1.221× | ✓ | **✓ best** |
| C | add_triton_autotune | 1.001× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_transpo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286020 ns | 234240 ns | -51780 ns (-18.1%) | 286620 ns | +600 ns (+0.2%) |
| SM Throughput (%) | 19.58 % | 43.98 % | +24.40 pp | 19.97 % | +0.39 pp |
| Memory Throughput (%) | 78.82 % | 53.01 % | -25.81 pp | 78.71 % | -0.11 pp |
| DRAM Throughput (%) | 72.15 % | 21.55 % | -50.60 pp | 72.00 % | -0.15 pp |
| Achieved Occupancy (%) | 94.02 % | 94.11 % | +0.09 pp | 94.06 % | +0.04 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.85 % | 62.43 % | -17.42 pp | 80.20 % | +0.35 pp |
| IPC Active | 0.81 inst/cycle | 1.51 inst/cycle | +0.70 | 0.79 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  286020 ns  (1.000×)
Iter 1     234240 ns  (1.221×)  — coalesced_transpose_with_tl_trans_and_autotune

Final best: 234240 ns  (1.2211×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=4),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=3),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=4),
+    triton.Config({'YBLOCK': 8,  'XBLOCK': 128}, num_warps=4, num_stages=2),
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
+    pid_x = tl.program_id(0)
+    pid_y = tl.program_id(1)
+    # rows: flat (b*128+c) index, size YBLOCK
+    rows = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
+    # cols: flat spatial (h*250+w) index, size XBLOCK
+    cols = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
+    row_mask = rows < ynumel
+    col_mask = cols < xnumel
+    c = rows % 128   # channel index (stride-1 in src)
+    b = rows // 128  # batch index
+    # Load [XBLOCK, YBLOCK]: threads span YBLOCK (channel, stride 1 in src) => COALESCED
+    src = c[None, :] + 128 * cols[:, None] + 4032000 * b[None, :]  # [XBLOCK, YBLOCK]
+    load_mask = col_mask[:, None] & row_mask[None, :]
+    tile = tl.load(in_ptr0 + src, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK]
+    # Store [YBLOCK, XBLOCK]: threads span XBLOCK (cols, stride 1 in dst) => COALESCED
+    dst = cols[None, :] + 31500 * rows[:, None]  # [YBLOCK, XBLOCK]
+    store_mask = row_mask[:, None] & col_mask[None, :]
+    tl.store(out_ptr0 + dst, tl.trans(tile), mask=store_mask)  # tl.trans: [XBLOCK,YBLOCK]->[YBLOCK,XBLOCK]
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

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
