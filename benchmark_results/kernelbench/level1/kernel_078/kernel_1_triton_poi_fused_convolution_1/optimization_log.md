# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285090 ns — **Best latency**: 285090 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_078/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285090 ns
- **SM Throughput**: 19.58 % | **Memory**: 79.06 % | **DRAM**: 72.36 %
- **Achieved Occupancy**: 93.70 %
- **IPC Active**: 0.81 inst/cycle | **No Eligible**: 79.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_tile_dimensions | 0.983× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285090 ns | 289660 ns | +4570 ns (+1.6%) |
| SM Throughput (%) | 19.58 % | 18.32 % | -1.26 pp |
| Memory Throughput (%) | 79.06 % | 77.94 % | -1.12 pp |
| DRAM Throughput (%) | 72.36 % | 71.25 % | -1.11 pp |
| Achieved Occupancy (%) | 93.70 % | 93.52 % | -0.18 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 79.77 % | 81.17 % | +1.40 pp |
| IPC Active | 0.81 inst/cycle | 0.75 inst/cycle | -0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_on_contiguous_y0_dimension | 0.984× | 0.984× | ✓ |  |
| D | increase_yblock_and_num_stages_for_l2_latency_hiding | 0.983× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_yblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285090 ns | 289700 ns | +4610 ns (+1.6%) | 288740 ns | +3650 ns (+1.3%) |
| SM Throughput (%) | 19.58 % | 18.61 % | -0.97 pp | 18.57 % | -1.01 pp |
| Memory Throughput (%) | 79.06 % | 77.87 % | -1.19 pp | 78.22 % | -0.84 pp |
| DRAM Throughput (%) | 72.36 % | 71.23 % | -1.13 pp | 71.48 % | -0.88 pp |
| Achieved Occupancy (%) | 93.70 % | 93.93 % | +0.23 pp | 93.96 % | +0.26 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.77 % | 81.35 % | +1.58 pp | 81.24 % | +1.47 pp |
| IPC Active | 0.81 inst/cycle | 0.74 inst/cycle | -0.07 | 0.73 inst/cycle | -0.08 |

---

## Cumulative Progress

```
Baseline:  285090 ns  (1.000×)

Final best: 285090 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +20,8 @@

     y0 = (yindex % 128)
     y1 = yindex // 128
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask & ymask)
+    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
