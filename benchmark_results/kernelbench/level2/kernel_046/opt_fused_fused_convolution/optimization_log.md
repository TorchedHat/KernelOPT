# Kernel Optimization Log

## Summary

- **Kernel**: `conv_epilogue_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16680000 ns — **Best latency**: 16680000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv_epilogue_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_046/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15876,1,1), Waves/SM: 60.14
- **Registers/thread**: 254
- **Duration**: 16680000 ns
- **SM Throughput**: 43.15 % | **Memory**: 59.91 % | **DRAM**: 2.56 %
- **Achieved Occupancy**: 12.49 %
- **IPC Active**: 1.74 inst/cycle | **No Eligible**: 56.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_k_decode_order_and_cap_registers | 0.947× | 0.946× | ✓ |  |
| C | eviction_policies_streaming_loads | 1.000× | 1.001× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_configs | 0.554× | 0.293× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_k_decode_orde... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 16680000 ns | 17630000 ns | +950000 ns (+5.7%) | 16660000 ns | -20000 ns (-0.1%) | 56830000 ns | +40150000 ns (+240.7%) |
| SM Throughput (%) | 43.15 % | 50.30 % | +7.15 pp | 43.12 % | -0.03 pp | 18.14 % | -25.01 pp |
| Memory Throughput (%) | 59.91 % | 55.36 % | -4.55 pp | 59.87 % | -0.04 pp | 95.51 % | +35.60 pp |
| DRAM Throughput (%) | 2.56 % | 2.45 % | -0.11 pp | 2.56 % | +0.00 pp | 1.14 % | -1.42 pp |
| Achieved Occupancy (%) | 12.49 % | 12.47 % | -0.02 pp | 12.49 % | +0.00 pp | 37.40 % | +24.91 pp |
| Registers/Thread | 254.00 register/thread | 255.00 register/thread | +1 | 254.00 register/thread | +0 | 80.00 register/thread | -174 |
| Waves/SM | 60.14 | 60.14 | +0.00 | 60.14 | +0.00 | 160.36 | +100.22 |
| No Eligible (%) | 56.48 % | 49.44 % | -7.04 pp | 56.55 % | +0.07 pp | 81.85 % | +25.37 pp |
| IPC Active | 1.74 inst/cycle | 2.02 inst/cycle | +0.28 | 1.74 inst/cycle | +0.00 | 0.73 inst/cycle | -1.01 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_via_smaller_tile_configs` (0.293×)

Reducing BLOCK_M×BLOCK_N tile sizes (e.g. from 256×128 to 64×64 or smaller) to lower register usage and improve occupancy can severely worsen performance when the kernel has significant shared memory bank conflicts. In this case, occupancy improved from 12.5% to 37.5% and registers dropped from 254 to 80 per thread, but Warp Cycles Per Issued Instruction exploded from 4.6 to 33.0 cycles and Mem Busy jumped from 60% to 95.5%, indicating that smaller tiles dramatically increased the ratio of shared memory accesses per output element, amplifying bank conflicts (from 5.3-way to 17.0-way average) and MIO pipeline pressure. The net effect was a 3.4× slowdown: the occupancy gain was completely overwhelmed by the increased per-element shared memory traffic and bank conflict stalls.

> **Applicable when**: Kernel has very high register usage (>200 registers/thread) causing occupancy <15%, AND shared memory bank conflicts are low (<5-way average), AND the bottleneck is truly register-limited occupancy rather than memory access pattern.

> **Avoid when**: The kernel already has significant shared memory bank conflicts (>5-way average, UncoalescedSharedAccess speedup_pct >30%). Smaller tiles increase the number of shared memory load/store operations per output element, amplifying bank conflicts and MIO pipeline pressure far more than the occupancy gain can compensate. Also avoid when Warp Cycles Per Issued Instruction is already low (<6), indicating the existing tile size is efficiently hiding latency.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_k_decode_order_for_coalesced_memory_access | 0.570× | 0.296× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_k_decode_orde... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 16680000 ns | 56400000 ns | +39720000 ns (+238.1%) |
| SM Throughput (%) | 43.15 % | 22.90 % | -20.25 pp |
| Memory Throughput (%) | 59.91 % | 95.04 % | +35.13 pp |
| DRAM Throughput (%) | 2.56 % | 1.15 % | -1.41 pp |
| Achieved Occupancy (%) | 12.49 % | 43.64 % | +31.15 pp |
| Registers/Thread | 254.00 register/thread | 72.00 register/thread | -182 |
| Waves/SM | 60.14 | 137.45 | +77.31 |
| No Eligible (%) | 56.48 % | 77.09 % | +20.61 pp |
| IPC Active | 1.74 inst/cycle | 0.92 inst/cycle | -0.82 |


### Key Learnings

#### ❌ Plan C: `fix_k_decode_order_for_coalesced_memory_access` (0.296×)

Reordering the K-dimension decode in a Triton convolution kernel to align the fastest-varying k-offset with the weight tensor's stride-1 dimension (kw) did not improve performance; instead it caused a 3.4× slowdown. The change was confounded by the autotuner simultaneously selecting much smaller tile sizes (BLOCK_M×BLOCK_N dropped from 256×64 to 64×64 or smaller), which dramatically increased shared memory traffic per output element and worsened bank conflicts from 5.3-way to 17.0-way average, causing Warp Cycles Per Issued Instruction to explode from 4.6 to 30.5 and Mem Busy to saturate at 95%. The K-decode reorder itself cannot be evaluated in isolation because the autotuner config space changed simultaneously, making it impossible to attribute the regression to the decode order change versus the tile size reduction.

> **Applicable when**: Never apply this specific combination: K-decode reorder paired with simultaneous autotuner config space reduction to smaller tiles. If testing K-decode reorder, hold tile sizes constant to isolate the effect.

> **Avoid when**: When the autotuner config space is changed simultaneously with the K-decode reorder — the two changes interact destructively. The tile size reduction dominates and amplifies shared memory bank conflicts far beyond any coalescing benefit from the decode reorder. Also do not apply when the kernel already has significant bank conflicts (>5-way) that are tile-size-sensitive.

---

## Cumulative Progress

```
Baseline:  16680000 ns  (1.000×)

Final best: 16680000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 @triton.autotune(
     configs=[
@@ -64,16 +65,16 @@

         valid = valid_m[:, None] & valid_k[None, :] & (batch[:, None] < BATCH)
         
         x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
-        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
+        x_tile = tl.load(x_ptrs, mask=valid, other=0.0, eviction_policy='evict_first')
         
         valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
         w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
-        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
+        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0, eviction_policy='evict_first')
         
         acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
     
     # Add bias
-    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
+    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0, eviction_policy='evict_last')
     acc = acc + bias[None, :]
     
     # sub 0.5, tanh, sub 0.2
@@ -84,7 +85,7 @@

     # Store output: shape (BATCH, OUT_C, OUT_H, OUT_W)
     mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
     out_ptrs = OUT_ptr + batch[:, None] * stride_on + offs_n[None, :] * stride_oc + out_h[:, None] * stride_oh + out_w[:, None] * stride_ow
-    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)
+    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out, eviction_policy='evict_first')
 
 
 @triton.autotune(
@@ -97,6 +98,7 @@

     key=['N_TOTAL'],
 )
 @triton.jit
+
 def avgpool2d_kernel(
     IN_ptr, OUT_ptr,
     BATCH: tl.constexpr, C: tl.constexpr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
