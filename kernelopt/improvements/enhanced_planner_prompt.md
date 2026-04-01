# Enhanced Planner System Prompt

## Additional Section to Add After "Step 2 — Cross-reference with raw metrics"

### Advanced Optimization Patterns

#### H100/H200 (Hopper) Architecture Specific Optimizations

**When gpu_specs.gpu_model contains "H100":**

1. **Async Copy Pipeline (num_stages)**
   - For memory-bound kernels with regular access patterns
   - Set `num_stages=3` or `num_stages=4` to enable TMA (Tensor Memory Accelerator)
   - Effective when: Memory Throughput > 40% AND SM Throughput < 60%
   - Example: `@triton.autotune(num_stages=[2, 3, 4])`

2. **Thread Block Cluster (TBC)**
   - New in Hopper: launch with `cluster_dims=(x, y, z)` for distributed shared memory
   - Use when: kernel needs >48KB shared memory OR benefits from cross-block communication
   - Check: `Dynamic Shared Memory` metric in profiling_context

3. **Warp Specialization**
   - Split warps into producer (load) and consumer (compute) roles
   - Use when: High "Warp Stall" percentage (>40%)
   - Implementation: Use `tl.program_id(1)` for warp role selection

4. **L2 Compression**
   - Hopper has hardware compression for L2 cache
   - NCU rule: "MemoryL2Compression" with speedup potential
   - Enable with specific data types (FP16, BF16, INT8) and aligned access

5. **Increased Register File**
   - H100 has 65536 registers per SM (vs 65536 in A100)
   - Can support higher occupancy with register-heavy kernels
   - Target: Increase `num_warps` if `Block Limit Registers` is NOT the constraint

#### Memory Access Pattern Optimizations

**When "Memory Throughput" > "SM Throughput" (memory-bound):**

1. **Vectorized Loads**
   - Use 128-bit loads (float4) instead of 32-bit loads
   - Check alignment: pointers should be 16-byte aligned
   - Triton: `tl.load(..., eviction_policy="evict_last")` for streaming data
   - CUDA: `__ldg()` for read-only data

2. **Cache Hierarchy Tuning**
   - L1 hit rate low (<70%): Increase tile size to improve locality
   - L2 hit rate low (<50%): Reduce working set or reorder computations
   - DRAM throughput high (>30%): Critical path is HBM bandwidth

3. **Shared Memory Banking**
   - NCU rule: "SharedMemoryBankConflicts"
   - For Triton: Pad shared arrays to avoid bank conflicts
   - Check: `Memory Workload Analysis -> Shared Memory Bank Conflicts`

4. **Async Barriers**
   - Use `tl.async_wait()` for fine-grained pipeline control
   - Overlap compute with memory fetch
   - Pattern: Load next tile while computing current tile

#### Compute Optimizations

**When "SM Throughput" > "Memory Throughput" (compute-bound):**

1. **Instruction Fusion**
   - Fuse operations: `tl.exp(x - row_max)` instead of separate subtract + exp
   - Use `1.0 / denom` followed by multiply instead of divide
   - Check ILP (Instruction-Level Parallelism) in NCU

2. **Fast Math**
   - For Triton: relies on LLVM optimizations automatically
   - For CUDA: `--use_fast_math` or `__fmul_rn()` intrinsics
   - Trade accuracy for speed (±1 ULP)

3. **Tensor Core Utilization**
   - For matmul/conv: ensure dimensions are multiples of 16 (FP16) or 8 (TF32)
   - Triton: `tl.dot()` with proper tile sizes (M%16==0, N%16==0, K%16==0)
   - Check: `Compute (SM) Throughput` should approach 100% for tensor ops

4. **Loop Unrolling**
   - Triton: `tl.constexpr` parameters are automatically unrolled
   - Increase `BLOCK_K` for reduction loops to amortize loop overhead
   - Trade: register pressure vs loop overhead

#### Occupancy & Scheduling

**When "Achieved Occupancy" < 70%:**

1. **Identify Limiter**
   - `Block Limit Registers`: Reduce tile size or use register spilling
   - `Block Limit Shared Memory`: Reduce SMEM allocation
   - `Block Limit Warps`: Increase block size
   - `Block Limit Blocks`: Grid size too small for GPU

2. **Warp Scheduling**
   - High "No Eligible Warps %": Memory latency not hidden
   - Solution: Increase occupancy OR use async copy
   - Target: Keep 4+ warps eligible per scheduler

3. **Wave Quantization**
   - NCU rule: "LaunchConfiguration" with partial wave tail
   - Grid should be multiple of (SM_count * max_blocks_per_SM)
   - For H100 SXM5: 132 SMs, typically 4-8 blocks/SM
   - Optimal grid: 132, 264, 528, 1056, ...

#### Advanced Triton-Specific

1. **Autotune Config Space**
   - Generate diverse configs: vary BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages
   - Use `@triton.autotune(configs=[...], key=['M', 'N', 'K'])`
   - Example configs for softmax:
     ```python
     configs=[
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
         triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
     ]
     ```

2. **Persistent Kernels**
   - For small grids (<132 on H100): launch once, loop internally
   - Reduces launch overhead, improves GPU utilization
   - Pattern: `while pid < total_work: process(pid); pid += num_programs`

3. **Split-K Parallelization**
   - For reductions: split along reduction dimension
   - Launch more blocks to increase parallelism
   - Atomic add for final reduction

#### Pattern Recognition from NCU Rules

**Rule: "TheoreticalOccupancy"**
- Meaning: Occupancy limited by resources
- Action: Reduce registers (smaller tile) OR reduce SMEM OR increase block size

**Rule: "CPIStall"**
- Meaning: Cycles per instruction is high due to stalls
- Common causes: Memory latency, dependency chains
- Action: Increase occupancy to hide latency OR async copy

**Rule: "IssueSlotUtilization"**
- Meaning: Not enough work to keep issue slots busy
- Action: Increase ILP (more independent operations) OR fuse operations

**Rule: "UncoalescedGlobalAccess"**
- Meaning: Non-contiguous memory access
- Action: Reorder data OR transpose layout OR use shared memory staging

**Rule: "WarpBranchEfficiency"**
- Meaning: Warp divergence from conditional branches
- Action: Remove branches OR predicated execution OR warp-uniform branches

#### Counterintuitive Optimizations (Learn from Memory)

1. **Smaller Tiles Can Be Faster**
   - When register pressure is high, smaller tiles reduce spilling
   - Trade: Less work per block vs better occupancy

2. **More Warps Not Always Better**
   - Diminishing returns beyond ~8 warps for simple kernels
   - Cache pollution from too many concurrent blocks

3. **Padding Inputs Can Help**
   - Round up dimensions to avoid partial wave tail
   - Trade: Extra memory reads vs better SM utilization

4. **Two-Pass Can Beat One-Pass**
   - Reduce register pressure by splitting kernel
   - Trade: Extra memory traffic vs better occupancy

---

## Enhanced Output Format

Add these fields to the plan JSON:

```json
{
  "plan_id": "...",
  "direction": "...",
  "framework": "...",
  
  "evidence": { ... },
  
  "step": {
    "action": "...",
    "target": "...",
    "change": "...",
    "predicted_effect": "...",
    
    // NEW: Implementation hints
    "implementation_hints": [
      "For Triton: Add @triton.autotune decorator with num_stages=[2,3,4]",
      "Measure: Memory Throughput should decrease by 10-15%"
    ],
    
    // NEW: Verification criteria
    "success_criteria": {
      "primary_metric": "Memory Throughput",
      "expected_direction": "decrease",
      "expected_magnitude": "10-20%",
      "acceptable_tradeoff": "SM Throughput may increase slightly"
    }
  },
  
  "correctness_constraints": [ ... ],
  
  // NEW: Architecture-specific notes
  "architecture_notes": "Hopper's TMA (Tensor Memory Accelerator) handles async copy with num_stages parameter",
  
  "memory_item_used": "...",
  "confidence": "...",
  "confidence_reason": "..."
}
```

---

## Reasoning Framework Enhancement

### Multi-Step Reasoning Chain

Instead of single hypothesis, use this chain:

1. **Identify Bottleneck** (memory vs compute)
2. **Quantify Headroom** (how far from peak?)
3. **Root Cause** (why the bottleneck exists?)
4. **Architecture Match** (does GPU have specific features?)
5. **Select Pattern** (from knowledge base above)
6. **Estimate Impact** (back-of-envelope calculation)
7. **Check Constraints** (correctness, memory limits)
8. **Generate Plan** (concrete implementation)

### Example Reasoning:


Headroom: 46% below peak (100% - 54%)
Root Cause: DRAM Throughput 32% indicates HBM bandwidth limit
Architecture: H100 SXM5 has 3350 GB/s peak, TMA async copy
Pattern Match: Async copy pipeline (num_stages) can hide latency
Estimate: With 4-stage pipeline, can hide ~75% of memory latency
  → Memory Throughput may stay same but SM Throughput increases
  → Overall kernel time reduces by ~10-15%
Constraint: Requires Hopper GPU ✓, Triton framework ✓
Plan: Add num_stages=4 to kernel launch
```
