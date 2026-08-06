# Kernel Optimization Log

## Summary

- **Kernel**: `conv2d_mish_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4260000 ns — **Best latency**: 3620000 ns (**1.1768×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_libdevice_tanh_with_tl_math_polynomial_approximation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv2d_mish_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_087/synthesis/kernel_0_fused_fused_convolution.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64516,1,1), Waves/SM: 244.38
- **Registers/thread**: 214
- **Duration**: 4260000 ns
- **SM Throughput**: 45.62 % | **Memory**: 19.82 % | **DRAM**: 11.24 %
- **Achieved Occupancy**: 12.37 %
- **IPC Active**: 1.83 inst/cycle | **No Eligible**: 54.35 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.019× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_configs | 0.998× | 1.087× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4260000 ns | 4180000 ns | -80000 ns (-1.9%) | 3920000 ns | -340000 ns (-8.0%) |
| SM Throughput (%) | 45.62 % | 45.55 % | -0.07 pp | 64.45 % | +18.83 pp |
| Memory Throughput (%) | 19.82 % | 20.39 % | +0.57 pp | 43.65 % | +23.83 pp |
| DRAM Throughput (%) | 11.24 % | 11.48 % | +0.24 pp | 13.64 % | +2.40 pp |
| Achieved Occupancy (%) | 12.37 % | 12.37 % | +0.00 pp | 55.07 % | +42.70 pp |
| Registers/Thread | 214.00 register/thread | 214.00 register/thread | +0 | 56.00 register/thread | -158 |
| Waves/SM | 244.38 | 244.38 | +0.00 | 217.23 | -27.15 |
| No Eligible (%) | 54.35 % | 54.25 % | -0.10 pp | 35.35 % | -19.00 pp |
| IPC Active | 1.83 inst/cycle | 1.82 inst/cycle | -0.01 | 2.59 inst/cycle | +0.76 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_for_l1tex_latency_hiding | 1.069× | 1.087× | ✓ | **✓ best** |
| B | hoist_k_index_arithmetic_outside_loop | 0.995× | 1.026× | ✓ |  |
| D | hoist_k_index_arithmetic_outside_loop | 1.029× | 1.148× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | hoist_k_index_ari... | Δ | hoist_k_index_ari... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260000 ns | 3920000 ns | -340000 ns (-8.0%) | 4150000 ns | -110000 ns (-2.6%) | 3710000 ns | -550000 ns (-12.9%) |
| SM Throughput (%) | 45.62 % | 64.44 % | +18.82 pp | 60.89 % | +15.27 pp | 60.96 % | +15.34 pp |
| Memory Throughput (%) | 19.82 % | 43.63 % | +23.81 pp | 41.46 % | +21.64 pp | 92.78 % | +72.96 pp |
| DRAM Throughput (%) | 11.24 % | 13.63 % | +2.39 pp | 12.89 % | +1.65 pp | 14.41 % | +3.17 pp |
| Achieved Occupancy (%) | 12.37 % | 55.04 % | +42.67 pp | 55.12 % | +42.75 pp | 42.95 % | +30.58 pp |
| Registers/Thread | 214.00 register/thread | 56.00 register/thread | -158 | 56.00 register/thread | -158 | 72.00 register/thread | -142 |
| Waves/SM | 244.38 | 217.23 | -27.15 | 217.23 | -27.15 | 279.29 | +34.91 |
| No Eligible (%) | 54.35 % | 35.32 % | -19.03 pp | 38.96 % | -15.39 pp | 38.98 % | -15.37 pp |
| IPC Active | 1.83 inst/cycle | 2.58 inst/cycle | +0.75 | 2.44 inst/cycle | +0.61 | 2.44 inst/cycle | +0.61 |


### Key Learnings

#### ✅ Plan A: `increase_num_stages_for_l1tex_latency_hiding` (1.087×)

When a Triton kernel shows significant L1TEX scoreboard stalls (CPIStall rule triggered, ~5 stall cycles per warp accounting for ~37% of CPI) with a very low L1 hit rate (~1%) but high L2 hit rate (~95%), adding autotune configs with num_stages=4 and num_stages=5 is the canonical approach to hide L2 load latency via deeper software pipelining. However, when the K-loop iteration count is very short (e.g. only 5 K-iterations for TOTAL_K=72, BLOCK_K=16), Triton's pipeliner may not generate effective async prefetch for such short loops, and the benefit is marginal — in this case, all key metrics (CPI, No-Eligible %, L2 hit rate, compute throughput, duration) remained essentially unchanged despite the config expansion. The speedup observed (1.087x) appears to be attributable to the prior small-tile optimization already captured in iteration 1, not to the num_stages increase itself.

> **Applicable when**: Triton matmul-style kernel with CPIStall rule triggered (L1TEX stall cycles >4 per warp, >30% of CPI); L1 hit rate <5%; L2 hit rate >80%; current max num_stages <=3; K-loop has many iterations (TOTAL_K / BLOCK_K >> 5) so the pipeline has room to overlap loads with compute.

> **Avoid when**: K-loop iteration count is very short (e.g. <=5 iterations as in TOTAL_K=72, BLOCK_K=16); in this regime Triton's pipeliner cannot effectively overlap loads across iterations and the CPIStall metrics do not improve. Also avoid when register pressure is already high (registers_per_thread near block limit) as deeper stages consume more registers for prefetch buffers.

#### ✅ Plan B: `hoist_k_index_arithmetic_outside_loop` (1.148×)

When a convolution K-loop decodes (kh, kw, in_c) from a flat index using vector integer division and modulo on BLOCK_K-wide tensors, replacing it with a triple nested loop (kh, kw, ic_start) makes kh and kw scalar compile-time constants and in_c a simple sequential range, eliminating all vector div/mod from the hot path. In this case, the restructuring raised L1/TEX hit rate from 1.06% to 64.79% (sequential in_c access becomes cache-friendly), reduced warp cycles per issued instruction from 13.64 to 11.25, and eliminated the SharedMemoryConflicts and UncoalescedSharedAccess rules entirely, yielding a 1.15x speedup despite a slight increase in registers/thread (56→72) and a drop in theoretical occupancy (56%→44%). The key insight is that sequential, predictable memory access patterns can outweigh occupancy gains when L1 hit rate improves dramatically.

> **Applicable when**: Triton convolution kernel uses a flat K-loop that decodes (kh, kw, in_c) from a flat index via vector integer division/modulo; L1/TEX hit rate is very low (<5%); SharedMemoryConflicts or UncoalescedSharedAccess rules are triggered; KH and KW are tl.constexpr enabling tl.static_range unrolling; CPIStall is significant (>30% of cycles stalled on L1TEX).

> **Avoid when**: KH and KW are not constexpr (tl.static_range requires constexpr bounds); the kernel is already using a nested loop structure; L1 hit rate is already high (>50%); the unrolling of KH*KW iterations would cause excessive code size bloat (e.g. KH*KW > 25); occupancy is the primary bottleneck and the register increase from unrolling would critically reduce block limit.

#### ✅ Plan D: `hoist_k_index_arithmetic_outside_loop` (1.148×)

When a convolution K-loop decodes (kh, kw, in_c) from a flat index using vector integer division and modulo on BLOCK_K-wide tensors, replacing it with a triple nested loop (kh, kw, ic_start) makes kh and kw scalar compile-time constants and in_c a simple sequential range, eliminating all vector div/mod from the hot path. In this case, the restructuring raised L1/TEX hit rate from 1.06% to 64.79% (sequential in_c access becomes cache-friendly), reduced warp cycles per issued instruction from 13.64 to 11.25, and eliminated the SharedMemoryConflicts and UncoalescedSharedAccess rules entirely, yielding a 1.15x speedup despite a slight increase in registers/thread (56→72) and a drop in theoretical occupancy (56%→44%). The key insight is that sequential, predictable memory access patterns can outweigh occupancy gains when L1 hit rate improves dramatically.

> **Applicable when**: Triton convolution kernel uses a flat K-loop that decodes (kh, kw, in_c) from a flat index via vector integer division/modulo; L1/TEX hit rate is very low (<5%); SharedMemoryConflicts or UncoalescedSharedAccess rules are triggered; KH and KW are tl.constexpr enabling tl.static_range unrolling; CPIStall is significant (>30% of cycles stalled on L1TEX).

> **Avoid when**: KH and KW are not constexpr (tl.static_range requires constexpr bounds); the kernel is already using a nested loop structure; L1 hit rate is already high (>50%); the unrolling of KH*KW iterations would cause excessive code size bloat (e.g. KH*KW > 25); occupancy is the primary bottleneck and the register increase from unrolling would critically reduce block limit.

---

## Iteration 3

**Best so far**: 1.0867× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | restructure_flat_k_loop_to_nested_kh_kw_ic_loops | 1.028× | 1.148× | ✓ |  |
| C | restructure_flat_k_loop_to_triple_nested_for_sequential_access | 1.028× | 1.148× | ✓ |  |
| D | add_triton_autotune | 1.029× | 1.148× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_flat_... | Δ | restructure_flat_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260000 ns | 3710000 ns | -550000 ns (-12.9%) | 3710000 ns | -550000 ns (-12.9%) | 3710000 ns | -550000 ns (-12.9%) |
| SM Throughput (%) | 45.62 % | 60.94 % | +15.32 pp | 60.94 % | +15.32 pp | 60.96 % | +15.34 pp |
| Memory Throughput (%) | 19.82 % | 92.79 % | +72.97 pp | 92.77 % | +72.95 pp | 92.79 % | +72.97 pp |
| DRAM Throughput (%) | 11.24 % | 14.41 % | +3.17 pp | 14.41 % | +3.17 pp | 14.41 % | +3.17 pp |
| Achieved Occupancy (%) | 12.37 % | 42.94 % | +30.57 pp | 42.94 % | +30.57 pp | 42.94 % | +30.57 pp |
| Registers/Thread | 214.00 register/thread | 72.00 register/thread | -142 | 72.00 register/thread | -142 | 72.00 register/thread | -142 |
| Waves/SM | 244.38 | 279.29 | +34.91 | 279.29 | +34.91 | 279.29 | +34.91 |
| No Eligible (%) | 54.35 % | 38.99 % | -15.36 pp | 38.97 % | -15.38 pp | 38.97 % | -15.38 pp |
| IPC Active | 1.83 inst/cycle | 2.44 inst/cycle | +0.61 | 2.44 inst/cycle | +0.61 | 2.44 inst/cycle | +0.61 |

---

## Iteration 4

**Best so far**: 1.0867× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | cap_registers_with_maxnreg_to_improve_occupancy | — | fail | ✓ |  |
| C | replace_libdevice_transcendentals_with_fused_fp32_intrinsics_in_mish_epilogue | 1.024× | 1.158× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4260000 ns | 3680000 ns | -580000 ns (-13.6%) |
| SM Throughput (%) | 45.62 % | 58.89 % | +13.27 pp |
| Memory Throughput (%) | 19.82 % | 92.77 % | +72.95 pp |
| DRAM Throughput (%) | 11.24 % | 14.54 % | +3.30 pp |
| Achieved Occupancy (%) | 12.37 % | 42.98 % | +30.61 pp |
| Registers/Thread | 214.00 register/thread | 72.00 register/thread | -142 |
| Waves/SM | 244.38 | 279.29 | +34.91 |
| No Eligible (%) | 54.35 % | 41.05 % | -13.30 pp |
| IPC Active | 1.83 inst/cycle | 2.36 inst/cycle | +0.53 |

---

## Iteration 5

**Best so far**: 1.1576× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_libdevice_tanh_with_tl_math_polynomial_approximation | 1.060× | 1.177× | ✓ |  |
| D | replace_libdevice_tanh_with_tl_math_exp_identity | 1.081× | 1.177× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ | replace_libdevice... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4260000 ns | 3620000 ns | -640000 ns (-15.0%) | 3620000 ns | -640000 ns (-15.0%) |
| SM Throughput (%) | 45.62 % | 53.99 % | +8.37 pp | 52.59 % | +6.97 pp |
| Memory Throughput (%) | 19.82 % | 94.90 % | +75.08 pp | 94.66 % | +74.84 pp |
| DRAM Throughput (%) | 11.24 % | 14.76 % | +3.52 pp | 14.76 % | +3.52 pp |
| Achieved Occupancy (%) | 12.37 % | 42.93 % | +30.56 pp | 42.89 % | +30.52 pp |
| Registers/Thread | 214.00 register/thread | 72.00 register/thread | -142 | 72.00 register/thread | -142 |
| Waves/SM | 244.38 | 279.29 | +34.91 | 279.29 | +34.91 |
| No Eligible (%) | 54.35 % | 45.96 % | -8.39 pp | 47.35 % | -7.00 pp |
| IPC Active | 1.83 inst/cycle | 2.16 inst/cycle | +0.33 | 2.11 inst/cycle | +0.28 |

---

## Cumulative Progress

```
Baseline:  4260000 ns  (1.000×)
Iter 2     3920000 ns  (1.087×)  — increase_num_stages_for_l1tex_latency_hiding
Iter 4     3680000 ns  (1.158×)  — replace_libdevice_transcendentals_with_fused_fp32_intrinsics_in_mish_epilogue
Iter 5     3620000 ns  (1.177×)  — replace_libdevice_tanh_with_tl_math_exp_identity

Final best: 3620000 ns  (1.1768×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,13 +4,23 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
+        # Small-tile configs — low register pressure, high occupancy
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
+        # ORIGINAL configs retained
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,'BLOCK_K': 16}, num_stages=2, num_warps=4),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
     ],
     key=['BATCH', 'OUT_C', 'OUT_H', 'OUT_W', 'IN_C'],
@@ -38,7 +48,6 @@

     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
 
     M_TOTAL = BATCH * OUT_H * OUT_W
-    TOTAL_K = KH * KW * IN_C
 
     # Decode M index -> (batch, out_h, out_w)
     out_w = offs_m % OUT_W
@@ -51,34 +60,34 @@

 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-    for k_start in range(0, TOTAL_K, BLOCK_K):
-        offs_k = k_start + tl.arange(0, BLOCK_K)
-        valid_k = offs_k < TOTAL_K
-
-        # Decode K index -> (kh, kw, in_c)
-        in_c = offs_k % IN_C
-        tmp_k = offs_k // IN_C
-        kw = tmp_k % KW
-        kh = tmp_k // KW
-
-        # Input spatial coordinates
-        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
-        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
-
-        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
-        mask_x = valid_m[:, None] & valid_k[None, :] & valid_spatial
-
-        x_ptrs = X_ptr + batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :] * s_xc
-        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
-
-        # Weight: (OUT_C, KH, KW, IN_C)
-        # offs_n: [BLOCK_N], offs_k: [BLOCK_K]
-        # w_ptrs shape: [BLOCK_K, BLOCK_N]
-        mask_w = valid_k[:, None] & valid_n[None, :]
-        w_ptrs = W_ptr + offs_n[None, :] * s_wn + kh[:, None] * s_wh + kw[:, None] * s_ww + in_c[:, None] * s_wc
-        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
-
-        acc = tl.dot(x_tile, w_tile, acc=acc)
+    # Triple nested loop: kh_i and kw_i are scalar constants (compile-time after unrolling),
+    # in_c is a simple sequential range — no vector div/mod needed.
+    for kh_i in tl.static_range(KH):          # KH=3 constexpr, fully unrolled
+        for kw_i in tl.static_range(KW):      # KW=3 constexpr, fully unrolled
+            # ih/iw computed once per (kh_i, kw_i) pair as 1D vectors [BLOCK_M]
+            ih = out_h * STRIDE_H + kh_i - PAD_H   # [BLOCK_M], scalar offset
+            iw = out_w * STRIDE_W + kw_i - PAD_W   # [BLOCK_M], scalar offset
+            # valid_spatial only depends on kh_i/kw_i — hoisted outside ic_start loop
+            valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)  # [BLOCK_M]
+            for ic_start in range(0, IN_C, BLOCK_K):
+                in_c = ic_start + tl.arange(0, BLOCK_K)  # sequential, no modulo
+                valid_k = in_c < IN_C
+                mask_x = valid_m[:, None] & valid_spatial[:, None] & valid_k[None, :]  # [BLOCK_M, BLOCK_K]
+                x_ptrs = (X_ptr
+                           + batch[:, None] * s_xn
+                           + ih[:, None]    * s_xh
+                           + iw[:, None]    * s_xw
+                           + in_c[None, :]  * s_xc)   # in_c is sequential
+                mask_w = valid_k[:, None] & valid_n[None, :]  # [BLOCK_K, BLOCK_N]
+                # kh_i*s_wh + kw_i*s_ww are compile-time constants after unrolling
+                w_ptrs = (W_ptr
+                           + offs_n[None, :] * s_wn
+                           + kh_i            * s_wh   # compile-time constant
+                           + kw_i            * s_ww   # compile-time constant
+                           + in_c[:, None]   * s_wc)  # [BLOCK_K, BLOCK_N]
+                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
+                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
+                acc = tl.dot(x_tile, w_tile, acc=acc)
 
     # Load bias and apply epilogue: bias + sub(0.5) + sub(0.2) + mish
     bias = tl.load(B_ptr + offs_n, mask=valid_n, other=0.0)
@@ -86,16 +95,27 @@

     acc = acc - SUB_VAL  # combined subtract: 0.5 + 0.2 = 0.7
 
     # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
-    # Numerically stable: if x > 20, softplus(x) ≈ x
-    softplus = tl.where(acc > 20.0, acc, libdevice.log(1.0 + libdevice.exp(acc)))
-    result = acc * libdevice.tanh(softplus)
+    # Numerically stable three-way split:
+    #   x > 20:  softplus(x) ≈ x  (exp overflow region)
+    #   x < -20: softplus(x) ≈ exp(x) ≈ 0  (exp underflow region)
+    #   else:    softplus(x) = log(1 + exp(x))
+    # Use tl.math.exp and tl.math.log (PTX hardware intrinsics, fuseable with FMA)
+    exp_acc = tl.math.exp(acc)                            # PTX ex2.approx.f32, fuseable
+    softplus_mid = tl.math.log(1.0 + exp_acc)             # PTX lg2.approx.f32, fuseable
+    # Three-way stable split: x>20 -> x, x<-20 -> exp(x)~0, else -> log1p(exp(x))
+    softplus = tl.where(acc > 20.0, acc,
+                   tl.where(acc < -20.0, exp_acc, softplus_mid))
+    # Inline tanh via identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
+    # All PTX ex2.approx.f32 intrinsics — FMA-fuseable, no opaque libdevice call boundary
+    exp2s = tl.math.exp(2.0 * softplus)       # PTX ex2.approx.f32, single SFU cycle
+    tanh_val = (exp2s - 1.0) / (exp2s + 1.0)  # 3 FP32 ops, compiler can FMA-fuse
+    result = acc * tanh_val
 
     # Store output in NCHW layout: (BATCH, OUT_C, OUT_H, OUT_W)
     # Linear index: batch * OUT_C * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w
     out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
     mask_out = valid_m[:, None] & valid_n[None, :]
     tl.store(Y_ptr + out_idx, result.to(tl.float32), mask=mask_out)
-
 
 def kernel_function(p_conv_weight, p_conv_bias, x):
     # x: (128, 8, 256, 256) NCHW
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
