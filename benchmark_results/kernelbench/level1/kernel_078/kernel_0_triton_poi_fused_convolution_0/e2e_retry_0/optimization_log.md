# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 4060 ns (**1.4261×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_fix_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_078/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 17.02 % | **Memory**: 42.67 % | **DRAM**: 21.28 %
- **Achieved Occupancy**: 55.27 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.84 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.371× | 1.426× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5790 ns | 4060 ns | -1730 ns (-29.9%) |
| SM Throughput (%) | 17.02 % | 19.32 % | +2.30 pp |
| Memory Throughput (%) | 42.67 % | 22.32 % | -20.35 pp |
| DRAM Throughput (%) | 21.28 % | 4.06 % | -17.22 pp |
| Achieved Occupancy (%) | 55.27 % | 48.37 % | -6.90 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.84 % | 56.15 % | -15.69 pp |
| IPC Active | 1.14 inst/cycle | 1.86 inst/cycle | +0.72 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_fix_uncoalesced_stores` (1.426×)

When a Triton pointwise kernel has high CPIStall from L1TEX (48% of warp cycles) and uncoalesced global stores (23% excessive sectors), adding @triton.autotune with configs exploring XBLOCK in [64,128,256,512,1024] and YBLOCK in [1,2,4,8] with num_stages=2 allows the runtime to select tile shapes that reduce stall cycles and L2 sector traffic. In this case, Warp Cycles Per Issued Instruction dropped from 29.87 to 16.3 and L2 Compression Input Sectors dropped from 260K to 75K, yielding a 1.43x speedup. The autotune approach is especially effective when the fixed tile shape was chosen without considering the specific memory access pattern of the kernel.

> **Applicable when**: Triton pointwise kernel with: (1) CPIStall rule triggered with L1TEX stall >40% of warp cycles, (2) UncoalescedGlobalAccess with >15% excessive sectors, (3) IssueSlotUtilization showing eligible warps <1.0 per scheduler, (4) fixed tile sizes hardcoded without autotuning. Kernel performs a layout transpose or strided store pattern.

> **Avoid when**: Kernel already uses @triton.autotune; kernel is compute-bound (SM Throughput >70%); grid is already large enough for >5 waves/SM with the fixed tile; adding autotune overhead is unacceptable in latency-critical single-call scenarios.

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     4060 ns  (1.426×)  — autotune_tile_sizes_fix_uncoalesced_stores

Final best: 4060 ns  (1.4261×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
