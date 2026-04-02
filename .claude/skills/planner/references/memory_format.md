# Optimization Memory Format

Schema and examples for experience items stored in the optimization memory queue.
Items are framework-agnostic — the same schema stores insights from CUDA, Triton, Helion, or any other framework.

---

## JSON schema

```json
{
  "item_id": "<uuid>",
  "iteration": <int>,
  "speedup": <float>,
  "rewrite_type": "positive" | "negative" | "implementation_flaw",
  "framework": "<cuda|triton|helion|unknown|multi>",
  "direction": "<snake_case, e.g. 'reduce_register_pressure'>",

  "profiling_signal": {
    "ncu_rule_triggered": "<Rule Name that prompted this plan, or null>",
    "key_metric_before": {
      "<Section Name>.<Metric Name>": <value>,
      ...
    },
    "key_metric_after": {
      "<Section Name>.<Metric Name>": <value>,
      ...
    }
  },

  "strategy_title": "<≤10 words>",
  "strategy_description": "<2-3 sentences, generalizable, references metric signals not variable names>",
  "slow_pseudocode": "<minimal before code>",
  "fast_pseudocode": "<minimal after code>",
  "applicable_when": "<profiling conditions that make this strategy relevant>",
  "do_not_apply_when": "<conditions where it causes regression or is inapplicable>",
  "framework_notes": "<framework-specific syntax notes, or null>"
}
```

---

## Positive example (Triton, register pressure)

```json
{
  "item_id": "e1a2b3c4",
  "iteration": 2,
  "speedup": 1.28,
  "rewrite_type": "positive",
  "framework": "triton",
  "direction": "reduce_tile_size_for_occupancy",

  "profiling_signal": {
    "ncu_rule_triggered": "TheoreticalOccupancy",
    "key_metric_before": {
      "Occupancy.Block Limit Registers": 5,
      "Occupancy.Theoretical Occupancy": 62.5,
      "Occupancy.Achieved Occupancy": 50.5,
      "Launch Statistics.Registers Per Thread": 44
    },
    "key_metric_after": {
      "Occupancy.Block Limit Registers": 10,
      "Occupancy.Theoretical Occupancy": 100.0,
      "Occupancy.Achieved Occupancy": 74.2,
      "Launch Statistics.Registers Per Thread": 22
    }
  },

  "strategy_title": "Halve tile size to reduce register pressure",
  "strategy_description": "When 'Occupancy → Block Limit Registers' is the binding occupancy constraint and registers per thread are high (>32), reducing the tile size cuts the number of live variables the compiler must track simultaneously, lowering register usage and raising theoretical occupancy. In Triton this means reducing BLOCK_K; in CUDA it means smaller shared memory tiles or `__launch_bounds__`.",
  "slow_pseudocode": "# BLOCK_K = 64 → registers_per_thread = 44, 5 blocks/SM\nfor k in range(0, K, BLOCK_K):\n    a = tl.load(a_ptr + k_offsets_64)\n    b = tl.load(b_ptr + k_offsets_64)\n    acc += tl.dot(a, b)",
  "fast_pseudocode": "# BLOCK_K = 32 → registers_per_thread = 22, 10 blocks/SM\nfor k in range(0, K, BLOCK_K):\n    a = tl.load(a_ptr + k_offsets_32)\n    b = tl.load(b_ptr + k_offsets_32)\n    acc += tl.dot(a, b)",
  "applicable_when": "Block Limit Registers < Block Limit Warps AND registers_per_thread > 32 AND TheoreticalOccupancy rule fires",
  "do_not_apply_when": "Kernel is already compute-bound (high SM throughput); reducing tile size increases loop iterations and may hurt more than it helps. Check that the tl.dot dimensions remain multiples of 16 after the change.",
  "framework_notes": "Triton: change BLOCK_K in @triton.autotune configs. CUDA: use __launch_bounds__(block_size, target_blocks_per_sm) or split the kernel."
}
```

---

## Implementation flaw example (Triton, invalid PTX modifier combination)

```json
{
  "item_id": "g9h0i1j2",
  "iteration": 2,
  "speedup": 0.0,
  "rewrite_type": "implementation_flaw",
  "framework": "triton",
  "direction": "enable_l2_compression",

  "profiling_signal": {
    "ncu_rule_triggered": "MemoryL2Compression",
    "key_metric_before": {
      "Memory Workload Analysis.DRAM Throughput": 48.1
    },
    "key_metric_after": {}
  },

  "strategy_title": "L2 eviction policy combined with wrong cache modifier",
  "strategy_description": "The direction (L2 eviction hints to improve cache reuse) is valid for memory-bound kernels. The implementation failed because tl.load was called with both cache_modifier='.cg' and eviction_policy='evict_last' simultaneously — ptxas rejects this combination on sm_90a. The correct approach is to use only one modifier: either cache_modifier='.cg' (cache in L2 globally) or tl.load with an explicit cache hint via inline PTX, not both.",
  "slow_pseudocode": "# WRONG: combining .cg and evict_last\nx = tl.load(ptr, cache_modifier='.cg', eviction_policy='evict_last')",
  "fast_pseudocode": "# CORRECT: use only one cache modifier\nx = tl.load(ptr, cache_modifier='.cg')",
  "applicable_when": "DRAM throughput > 40% on memory-bound kernels where L2 reuse is low",
  "do_not_apply_when": "Never combine cache_modifier='.cg' with eviction_policy='evict_last' — invalid on Hopper (sm_90a). Choose one.",
  "framework_notes": "Triton on sm_90a: cache_modifier and eviction_policy generate separate PTX modifiers. ptxas will reject .cg + .evict_last. Use cache_modifier='.cg' alone or eviction_policy='evict_last' alone."
}
```

---

## Negative example (CUDA, over-tiling caused spill)

```json
{
  "item_id": "f5d6e7a8",
  "iteration": 4,
  "speedup": 0.81,
  "rewrite_type": "negative",
  "framework": "cuda",
  "direction": "double_tile_size",

  "profiling_signal": {
    "ncu_rule_triggered": null,
    "key_metric_before": {
      "Memory Workload Analysis.Local Memory Spilling Requests": 0,
      "Occupancy.Achieved Occupancy": 50.5
    },
    "key_metric_after": {
      "Memory Workload Analysis.Local Memory Spilling Requests": 12800,
      "Occupancy.Achieved Occupancy": 24.1
    }
  },

  "strategy_title": "Doubling tile size caused register spill to local memory",
  "strategy_description": "Increasing the tile size doubled the number of live variables per thread, exceeding the register file budget. The compiler spilled registers to local memory (visible as Local Memory Spilling Requests jumping from 0 to 12800), which is slower than global memory access because it goes through L1. Occupancy also dropped because spill increases effective register usage per thread.",
  "slow_pseudocode": "// TILE=64: 0 spill requests, achieved occupancy 50%\n__shared__ float smem[64][64];",
  "fast_pseudocode": "// TILE=128: 12800 spill requests, achieved occupancy 24%\n__shared__ float smem[128][128];",
  "applicable_when": "N/A — this caused regression",
  "do_not_apply_when": "When registers_per_thread is already high (>32) or Block Limit Registers is the occupancy constraint. Doubling tile size in this regime will almost always cause spill.",
  "framework_notes": null
}
```

---

## Memory queue management

```json
// optimization_memory.json — persisted between iterations and sessions
{
  "expn": 8,
  "items": [
    { ...item... },
    { ...item... }
  ]
}
```

Rules:
- Append new items to the end; drop from the front when `len(items) > expn`
- At most 3 items per `direction` in the queue (prevents one strategy dominating)
- Negative items count against a separate budget (`topk // 2` slots)
- Persist to disk between sessions — the memory accumulates across multiple kernel runs
  and becomes more valuable over time
- The `framework` field is informational; planners should consider items from all frameworks
  since the underlying hardware signals are the same
