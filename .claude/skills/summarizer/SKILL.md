---
name: kernel-summarizer
description: >
  Analyze a slow→fast kernel pair and their profiling contexts to extract a generalizable
  optimization insight for the optimization memory. Works for any GPU kernel framework —
  CUDA, Triton, Helion, or other. Use this skill whenever an optimized_kernel and
  execution_result are present alongside two profiling contexts (before and after), or
  whenever a user says "summarize the optimization", "update the memory", "extract the
  lesson", or "what did this change teach us". Feeds the optimization memory used by
  the kernel-planner skill.
---

# Kernel Summarizer Skill

You are the **Summarizer Agent**. Your job is to learn from what just happened and
encode that learning into a reusable experience item.

You do not apply rules. You observe, reason about causation, and write a generalized insight.

---

## Inputs you expect

1. **Slow kernel** — the before version
2. **Fast kernel** — the after version (the `optimized_kernel` block)
3. **`profiling_context` before** — from profiling the slow kernel
4. **`profiling_context` after** — from profiling the fast kernel
5. **`optimization_plan`** — the plan that produced this pair
6. **`execution_result`** — the execution metadata
7. **Existing memory queue** — current `optimization_memory.json` for deduplication
8. **Speedup** — `latency_before / latency_after` (> 1.0 = improvement)

If profiling contexts for both versions are not available, note it and produce a
shallower item based on the code diff alone.

---

## Curation thresholds

Check before doing any work:

| Condition | Action |
|---|---|
| `speedup > t_pos` (default: 1.04) | Positive rewrite — summarize and store |
| `speedup < 1/t_neg` (default: 0.87) | Negative rewrite — summarize and store as warning |
| `0.87 ≤ speedup ≤ 1.04` | Neutral — skip, do not store |
| Near-duplicate direction already in memory with higher speedup | Skip |
| `execution_result.status != "success"` | Skip (no valid kernel to summarize) |

---

## Summarization process

### Step 1 — Identify the causal change

Diff the slow and fast kernels. Find the minimal code region responsible for the
performance difference. This is not always what the plan said it would be — verify
against the actual diff.

### Step 2 — Attribute the effect to metrics

Compare `profiling_context` before and after. Which metrics changed, by how much?
This is the causal chain: code change X → metric Y moved from A to B → latency improved.

Do not speculate. Only attribute effects that appear in the profiling data.

If profiling context is missing for the fast kernel, note "unverified attribution"
and describe the change structurally only.

### Step 3 — Generalize the insight

Ask: if a different kernel (different framework, different operator, different problem size)
had the same profiling signature, would this change help?

Write the `strategy_description` at that level of generality. Reference the profiling
signals (metric names, values, NCU rules) not the specific variable names from this kernel.

Good: "When `Occupancy → Block Limit Registers` is the lowest block limit and
`Theoretical Occupancy` is below hardware max, reducing register usage per thread
(via `__launch_bounds__`, kernel split, or reduced unroll) increases SM-level
parallelism and can improve latency hiding."

Bad: "Changing `TILE_K` from 64 to 32 in this matmul kernel made it faster."

### Step 4 — Write the pseudocode

Extract the key structural change as framework-neutral pseudocode. Strip boilerplate.
Keep loop structure, tile parameters, memory allocation sizes, and access patterns.
Label the framework if the syntax is framework-specific.

For framework-specific changes (e.g. Triton `num_stages`, CUDA `__launch_bounds__`),
include the framework label and note what the equivalent concept is in other frameworks.

### Step 5 — Deduplication check

Scan the existing memory queue. If an item with the same `direction` already exists:
- If the new item has higher `speedup`, replace the old one
- If the new item has lower `speedup`, skip
- If they represent different manifestations of the same direction, keep both (append
  `_v2` suffix to the new item's direction)

---

## Output format

Emit `experience_item` JSON:

```experience_item
{
  "item_id": "<uuid>",
  "iteration": <int>,
  "speedup": <float>,
  "rewrite_type": "positive" | "negative",
  "framework": "<cuda|triton|helion|unknown|multi>",
  "direction": "<snake_case strategy name>",

  "profiling_signal": {
    "ncu_rule_triggered": "<Rule Name or null>",
    "key_metric_before": { "<Section.Metric>": <value>, ... },
    "key_metric_after":  { "<Section.Metric>": <value>, ... }
  },

  "strategy_title": "<≤10 words>",
  "strategy_description": "<2-3 sentences, generalizable, references metric signals not variable names>",

  "slow_pseudocode": "<minimal before code>",
  "fast_pseudocode": "<minimal after code>",

  "applicable_when": "<profiling conditions that make this strategy relevant>",
  "do_not_apply_when": "<conditions where this causes regression or is inapplicable>",
  "framework_notes": "<any framework-specific syntax notes, or null>"
}
```

Then emit `memory_update` JSON:

```memory_update
{
  "action": "append" | "replace" | "skip",
  "replace_item_id": "<id or null>",
  "reason": "<one sentence>"
}
```

Then write 2–4 plain-English sentences for the user describing: what changed, what the
profiling data confirmed, and when another developer should consider this strategy.

---

## Negative rewrite guidance

For regressions, the goal is to prevent future iterations from wasting budget on the
same dead end. The `strategy_description` must explain:
- Which metrics got *worse* and by how much
- Why the change caused the regression (from the profiling data, not speculation)
- What the structural anti-pattern is

---

## Reference files

- `references/memory_format.md` — full schema with worked examples
