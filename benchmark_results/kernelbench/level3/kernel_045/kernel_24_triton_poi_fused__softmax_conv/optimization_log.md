# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2210 ns — **Best latency**: 2210 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_24_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 27
- **Duration**: 2210 ns
- **SM Throughput**: 0.02 % | **Memory**: 0.42 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.24 inst/cycle | **No Eligible**: 91.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_utilize_all_sms | — | 1.000× | ✓ | **✓ best** |
| B | fix_grid_launch_to_cover_y_dimension | — | 1.000× | ✓ |  |
| C | fix_grid_parallelism_yblock_xblock_swap | — | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_launch_t... | Δ | fix_grid_parallel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2210 ns | 2210 ns | +0 ns (+0.0%) | 2210 ns | +0 ns (+0.0%) | 2210 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.02 % | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.45 % | +0.03 pp | 0.42 % | +0.00 pp | 0.47 % | +0.05 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.21 % | 6.14 % | -0.07 pp | 6.12 % | -0.09 pp | 6.14 % | -0.07 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.51 % | 90.65 % | -0.86 pp | 91.97 % | +0.46 pp | 93.15 % | +1.64 pp |
| IPC Active | 0.24 inst/cycle | 0.36 inst/cycle | +0.12 | 0.36 inst/cycle | +0.12 | 0.36 inst/cycle | +0.12 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | vectorize_loads_stores_via_xblock_alignment | — | 0.921× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorize_loads_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2210 ns | 2400 ns | +190 ns (+8.6%) |
| SM Throughput (%) | 0.02 % | 0.02 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.39 % | -0.03 pp |
| DRAM Throughput (%) | 0.04 % | 0.03 % | -0.01 pp |
| Achieved Occupancy (%) | 6.21 % | 6.17 % | -0.04 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.51 % | 93.76 % | +2.25 pp |
| IPC Active | 0.24 inst/cycle | 0.34 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  2210 ns  (1.000×)

Final best: 2210 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -21,7 +22,6 @@

     y1 = yindex // 512
     tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 512*x2 + 2048*y1), tmp0, xmask & ymask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
