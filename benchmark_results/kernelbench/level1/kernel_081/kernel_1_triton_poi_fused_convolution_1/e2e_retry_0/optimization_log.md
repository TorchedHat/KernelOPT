# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285760 ns — **Best latency**: 319260 ns (**0.8951×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285760 ns
- **SM Throughput**: 19.98 % | **Memory**: 78.87 % | **DRAM**: 72.21 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | coalesced_tiled_transpose_with_autotune | 2.419× | 0.895× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | coalesced_tiled_t... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285760 ns | 319260 ns | +33500 ns (+11.7%) |
| SM Throughput (%) | 19.98 % | 22.26 % | +2.28 pp |
| Memory Throughput (%) | 78.87 % | 62.38 % | -16.49 pp |
| DRAM Throughput (%) | 72.21 % | 62.38 % | -9.83 pp |
| Achieved Occupancy (%) | 94.05 % | 94.90 % | +0.85 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.09 % | 78.52 % | -1.57 pp |
| IPC Active | 0.79 inst/cycle | 0.86 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  285760 ns  (1.000×)

Final best: 319260 ns  (0.8951×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
+    triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=4),
+    triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=4),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
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
+    pid_m = tl.program_id(0)  # over ynumel=4096 (channel/batch)
+    pid_n = tl.program_id(1)  # over xnumel=31500 (spatial)
+    m_offsets = pid_m * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
+    n_offsets = pid_n * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
+    y0 = m_offsets % 128   # channel index, stride-1 in source
+    y1 = m_offsets // 128  # batch index
+    x2 = n_offsets         # spatial index
+    m_mask = m_offsets < ynumel
+    n_mask = n_offsets < xnumel
+    load_mask = m_mask[:, None] & n_mask[None, :]
+    # Coalesced load: adjacent m-threads differ by 1 in y0 (stride-1 in source)
+    src_idx = y0[:, None] + 128 * x2[None, :] + 4032000 * y1[:, None]
+    tile = tl.load(in_ptr0 + src_idx, mask=load_mask, other=0.0, eviction_policy='evict_last')
+    # Coalesced store: adjacent n-threads differ by 1 in x2 (stride-1 in dest)
+    dst_idx = x2[None, :] + 31500 * m_offsets[:, None]
+    tl.store(out_ptr0 + dst_idx, tile, mask=load_mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +40,7 @@

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
