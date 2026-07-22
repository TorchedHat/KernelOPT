# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11360 ns — **Best latency**: 11520 ns (**0.9861×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_native_group_norm_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_088/kernel_1_triton_per_fused_native_group_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 32
- **Duration**: 11360 ns
- **SM Throughput**: 28.10 % | **Memory**: 62.43 % | **DRAM**: 62.43 %
- **Achieved Occupancy**: 82.49 %
- **IPC Active**: 1.40 inst/cycle | **No Eligible**: 64.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_transpose_tile_layout | 1.016× | 0.986× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 11360 ns | 11520 ns | +160 ns (+1.4%) |
| SM Throughput (%) | 28.10 % | 27.69 % | -0.41 pp |
| Memory Throughput (%) | 62.43 % | 61.68 % | -0.75 pp |
| DRAM Throughput (%) | 62.43 % | 61.68 % | -0.75 pp |
| Achieved Occupancy (%) | 82.49 % | 82.49 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 64.72 % | 64.62 % | -0.10 pp |
| IPC Active | 1.40 inst/cycle | 1.40 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  11360 ns  (1.000×)

Final best: 11520 ns  (0.9861×)
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
 def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
@@ -13,27 +14,26 @@

     rnumel = r0_numel
     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
+    r0_index = tl.arange(0, R0_BLOCK)[:, None]
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
-    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
+    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[:, None]
     roffset = r0_offset
     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0, eviction_policy='evict_first')
-    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
-    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
-    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
+    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, XBLOCK])
+    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK, XBLOCK])
+    tmp5 = tl.sum(tmp3, 0)[None, :].to(tl.float32)
     tmp8 = tmp5 * (1.0 / 16.0)
     tmp9 = tmp1 - tmp8
     tmp10 = tmp9 * tmp9
-    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
-    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
+    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK, XBLOCK])
+    tmp13 = tl.sum(tmp11, 0)[None, :].to(tl.float32)
     tl.store(out_ptr0 + (x0), tmp8, xmask, eviction_policy='evict_last')
     tl.store(out_ptr1 + (x0), tmp13, xmask, eviction_policy='evict_last')
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
