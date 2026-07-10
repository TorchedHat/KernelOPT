# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_mul_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 646050 ns — **Best latency**: 646050 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_mul_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_020/kernel_0_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1048576,1,1), Waves/SM: 496.48
- **Registers/thread**: 16
- **Duration**: 646050 ns
- **SM Throughput**: 32.77 % | **Memory**: 33.00 % | **DRAM**: 33.00 %
- **Achieved Occupancy**: 44.06 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 67.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_larger_xblock | 1.000× | 1.000× | ✓ |  |
| C | increase_xblock_via_autotune_to_hide_l1tex_latency | 0.997× | 0.309× | ✓ |  |
| D | increase_xblock_with_autotune | 0.996× | 0.312× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_xblock_v... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 646050 ns | 646210 ns | +160 ns (+0.0%) | 2090000 ns | +1443950 ns (+223.5%) | 2070000 ns | +1423950 ns (+220.4%) |
| SM Throughput (%) | 32.77 % | 32.78 % | +0.01 pp | 83.64 % | +50.87 pp | 83.45 % | +50.68 pp |
| Memory Throughput (%) | 33.00 % | 32.98 % | -0.02 pp | 43.07 % | +10.07 pp | 43.51 % | +10.51 pp |
| DRAM Throughput (%) | 33.00 % | 32.98 % | -0.02 pp | 10.22 % | -22.78 pp | 10.33 % | -22.67 pp |
| Achieved Occupancy (%) | 44.06 % | 44.06 % | +0.00 pp | 78.01 % | +33.95 pp | 72.85 % | +28.79 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 21.00 register/thread | +5 | 21.00 register/thread | +5 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 1985.94 | +1489.46 | 1985.94 | +1489.46 |
| No Eligible (%) | 67.09 % | 67.09 % | +0.00 pp | 16.46 % | -50.63 pp | 16.36 % | -50.73 pp |
| IPC Active | 1.32 inst/cycle | 1.32 inst/cycle | +0.00 | 3.34 inst/cycle | +2.02 | 3.34 inst/cycle | +2.02 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_via_autotune_to_hide_l1tex_latency` (0.309×)

Adding @triton.autotune with variable XBLOCK while keeping the launch grid as a static tuple computed with the old XBLOCK (e.g., cdiv(N, 128)) causes the grid to over-launch by a factor of new_XBLOCK/old_XBLOCK. When autotune selects XBLOCK=512, the grid remains at cdiv(N, 128)=4x too many blocks, causing each element to be processed 4x redundantly — inflating executed instructions from 168M to 1.38B (8.2x) and duration from 646µs to 2.09ms (3.2x regression). Per-scheduler metrics improved (eligible warps 0.39→7.4, compute throughput 33%→83%) but total work explosion dominated. The grid MUST be a lambda: `lambda meta: (triton.cdiv(N, meta['XBLOCK']),)`.

> **Applicable when**: Any Triton kernel where @triton.autotune is added with variable XBLOCK configs but the launch grid is a static tuple computed with the original (smaller) XBLOCK value. Symptom: executed instruction count increases by ~(new_XBLOCK/old_XBLOCK)x while DRAM throughput drops and compute throughput rises artificially.

> **Avoid when**: This is a negative/anti-pattern entry — never use a static grid when autotune varies XBLOCK. The fix (lambda grid) should always be applied when autotune is used with variable XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_with_correct_lambda_grid | 0.999× | 1.000× | ✓ |  |
| B | fix_autotune_grid_lambda_and_add_vectorized_load_hints | 0.997× | 0.312× | ✓ |  |
| C | fix_autotune_grid_lambda_and_vectorized_loads | 0.997× | 0.309× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_w... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 646050 ns | 646270 ns | +220 ns (+0.0%) | 2070000 ns | +1423950 ns (+220.4%) | 2090000 ns | +1443950 ns (+223.5%) |
| SM Throughput (%) | 32.77 % | 36.92 % | +4.15 pp | 83.55 % | +50.78 pp | 83.46 % | +50.69 pp |
| Memory Throughput (%) | 33.00 % | 32.98 % | -0.02 pp | 43.56 % | +10.56 pp | 42.97 % | +9.97 pp |
| DRAM Throughput (%) | 33.00 % | 32.98 % | -0.02 pp | 10.32 % | -22.68 pp | 10.22 % | -22.78 pp |
| Achieved Occupancy (%) | 44.06 % | 45.02 % | +0.96 pp | 72.50 % | +28.44 pp | 78.02 % | +33.96 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 21.00 register/thread | +5 | 21.00 register/thread | +5 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 1985.94 | +1489.46 | 1985.94 | +1489.46 |
| No Eligible (%) | 67.09 % | 62.94 % | -4.15 pp | 16.45 % | -50.64 pp | 16.39 % | -50.70 pp |
| IPC Active | 1.32 inst/cycle | 1.48 inst/cycle | +0.16 | 3.34 inst/cycle | +2.02 | 3.35 inst/cycle | +2.03 |

---

## Cumulative Progress

```
Baseline:  646050 ns  (1.000×)

Final best: 646050 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_add_convolution_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
     xnumel = 134217728
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:]
+    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[:], 16), 16)
     xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x3 = xindex
     x1 = ((xindex // 131072) % 64)
@@ -22,8 +23,6 @@

     tmp6 = tmp5 * tmp2
     tmp7 = tmp6 + tmp2
     tl.store(in_out_ptr0 + (x3), tmp7, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
