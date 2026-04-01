# Planner Confidence Calculation

## How is "confidence" calculated in optimization plans?

The `confidence` field in optimization plans is **not calculated algorithmically** — it is **reasoned by Claude** based on the strength of evidence and directness of the optimization path.

---

## Confidence Levels

```
"confidence": "high|medium|low"
```

Accompanied by:
```
"confidence_reason": "<explanation of why this confidence level>"
```

---

## What Claude Considers

### High Confidence ✅

**When:**
- NCU rule **directly identifies** the bottleneck
- Speedup percentage is **quantified** by NCU
- Optimization path is **clear and direct**
- Risk of side effects is **low**
- Change is **localized** and **well-understood**

**Examples from our tests:**

```json
{
  "confidence": "high",
  "confidence_reason": "NCU rule directly identifies 50% speedup opportunity from eliminating partial wave tail effect",
  "evidence": {
    "ncu_rule": "LaunchConfiguration",
    "ncu_rule_speedup_pct": 50.0
  }
}
```

```json
{
  "confidence": "high",
  "confidence_reason": "NCU rule directly identifies register pressure as the occupancy limiter with quantified 37.5% speedup potential",
  "evidence": {
    "ncu_rule": "TheoreticalOccupancy",
    "ncu_rule_speedup_pct": 37.5
  }
}
```

**Indicators:**
- ✅ NCU rule present with speedup %
- ✅ Clear cause-effect relationship
- ✅ "Directly identifies" language
- ✅ Quantified impact

---

### Medium Confidence ⚠️

**When:**
- NCU rule identifies the bottleneck but...
- **Risk of side effects** or trade-offs
- Optimization might **expose new bottlenecks**
- Change requires **more complex reasoning**
- Speedup is **conditional** on other factors

**Example from our tests:**

```json
{
  "confidence": "medium",
  "confidence_reason": "NCU rule directly identifies scheduler stalls as 33% bottleneck, but reducing warps risks exposing other latencies if memory-bound portions emerge",
  "evidence": {
    "ncu_rule": "CPIStall",
    "ncu_rule_speedup_pct": 33.43
  }
}
```

**Indicators:**
- ⚠️ NCU rule present BUT...
- ⚠️ "...but" or "...risks" in confidence_reason
- ⚠️ Mentions potential side effects
- ⚠️ Conditional speedup
- ⚠️ May expose other bottlenecks

---

### Low Confidence ⚠️⚠️

**When:**
- No direct NCU rule (inferring from indirect signals)
- Optimization is **speculative**
- Requires **multiple changes** to see benefit
- High risk of **breaking correctness**
- Impact is **hard to predict**

**Example scenarios:**

```json
{
  "confidence": "low",
  "confidence_reason": "Inferring from indirect signals - no specific NCU rule points to this, but cache metrics suggest alignment issues",
  "evidence": {
    "ncu_rule": null,
    "ncu_rule_speedup_pct": null
  }
}
```

**Indicators:**
- ⚠️⚠️ No NCU rule (`null`)
- ⚠️⚠️ "Inferring from indirect signal"
- ⚠️⚠️ Speculative optimization
- ⚠️⚠️ Uncertain impact

---

## Pattern Recognition

### Confidence = HIGH

```
NCU Rule → Clear Bottleneck → Direct Optimization → Quantified Impact
```

**Pattern in confidence_reason:**
- "NCU rule **directly** identifies..."
- "...with **quantified** X% speedup potential"
- "Clear cause-effect relationship"

### Confidence = MEDIUM

```
NCU Rule → Clear Bottleneck → Optimization → BUT Trade-offs/Risks
```

**Pattern in confidence_reason:**
- "NCU rule identifies... **but**..."
- "...risks exposing..."
- "...conditional on..."
- "...may affect..."

### Confidence = LOW

```
Indirect Signals → Hypothesis → Speculative Optimization → Unknown Impact
```

**Pattern in confidence_reason:**
- "**Inferring** from indirect signals"
- "No specific NCU rule"
- "Speculative based on..."
- "Uncertain impact"

---

## Real Examples from Tests

### Plan 1: High Confidence (50% speedup)

```json
{
  "direction": "reduce_partial_wave_tail_effect",
  "confidence": "high",
  "confidence_reason": "NCU rule directly identifies 50% speedup opportunity from eliminating partial wave tail effect",
  "evidence": {
    "ncu_rule": "LaunchConfiguration",
    "ncu_rule_speedup_pct": 50.0,
    "supporting_metrics": {
      "Launch Statistics.Grid Size": 1056.0,
      "Launch Statistics.# SMs": 132.0,
      "Launch Statistics.Waves Per SM": 1.6
    },
    "reasoning": "NCU reports 1 full wave plus partial wave of 397 blocks, causing up to 50% runtime overhead. Grid size 1056 on 132 SMs creates suboptimal wave distribution with significant tail effect."
  }
}
```

**Why high?**
- ✅ NCU LaunchConfiguration rule with 50% speedup
- ✅ Quantified problem: 397 wasted blocks
- ✅ Clear solution: align grid to SM count
- ✅ Direct cause-effect: eliminate partial wave → 50% faster

---

### Plan 2: High Confidence (37.5% speedup)

```json
{
  "direction": "reduce_register_pressure_for_occupancy",
  "confidence": "high",
  "confidence_reason": "NCU rule directly identifies register pressure as the occupancy limiter with quantified 37.5% speedup potential",
  "evidence": {
    "ncu_rule": "TheoreticalOccupancy",
    "ncu_rule_speedup_pct": 37.5,
    "reasoning": "Theoretical occupancy of 62.5% is limited by register usage (44 regs/thread). Reducing register pressure would allow more concurrent warps, improving throughput by up to 37.5%."
  }
}
```

**Why high?**
- ✅ NCU TheoreticalOccupancy rule with 37.5% speedup
- ✅ Clear limiter: registers per thread
- ✅ Well-known optimization: reduce register usage
- ✅ Predictable impact on occupancy

---

### Plan 3: Medium Confidence (33.43% speedup)

```json
{
  "direction": "reduce_warp_scheduler_stalls",
  "confidence": "medium",
  "confidence_reason": "NCU rule directly identifies scheduler stalls as 33% bottleneck, but reducing warps risks exposing other latencies if memory-bound portions emerge",
  "evidence": {
    "ncu_rule": "CPIStall",
    "ncu_rule_speedup_pct": 33.43,
    "reasoning": "Kernel spends 3.7 cycles per instruction in scheduler stalls. Increasing instruction-level parallelism through reordering could reduce stalls, though effect depends on whether compute or memory bound."
  }
}
```

**Why medium (not high)?**
- ⚠️ NCU rule present BUT solution has risks
- ⚠️ "...but reducing warps risks..."
- ⚠️ Conditional: "depends on whether compute or memory bound"
- ⚠️ May expose other bottlenecks

---

## Confidence Reasoning Flow

```
Claude analyzes profiling_context:
    ↓
Step 1: Is there an NCU rule?
    ↓ YES                          ↓ NO
    ↓                              ↓
Step 2: Is speedup quantified?    → confidence = LOW
    ↓ YES                         (inferring from indirect signals)
    ↓
Step 3: Is optimization path clear?
    ↓ YES                          ↓ NO
    ↓                              ↓
Step 4: Are there risks/trade-offs? → confidence = MEDIUM
    ↓ NO                          (complex or uncertain)
    ↓
confidence = HIGH
(direct, quantified, low-risk)
```

---

## Key Factors Claude Considers

### Evidence Strength
- **Strong:** NCU rule with speedup % → HIGH
- **Medium:** NCU rule with caveats → MEDIUM
- **Weak:** Indirect signals → LOW

### Optimization Clarity
- **Clear:** Single change, direct impact → HIGH
- **Complex:** Multiple changes needed → MEDIUM/LOW

### Risk Assessment
- **Low risk:** Localized change, well-understood → HIGH
- **Medium risk:** May expose new bottlenecks → MEDIUM
- **High risk:** Speculative, uncertain impact → LOW

### Predictability
- **Quantified impact:** NCU provides % → HIGH
- **Estimated impact:** Can calculate effect → MEDIUM
- **Unknown impact:** Hard to predict → LOW

---

## Summary

**Confidence is NOT a formula** — it's **Claude's reasoning** about:

1. **Evidence strength:** Direct NCU rule vs indirect signals
2. **Optimization clarity:** Clear path vs complex changes
3. **Risk level:** Low-risk local change vs high-risk speculation
4. **Impact predictability:** Quantified vs estimated vs unknown

**The `confidence_reason` field is key** — it explains exactly why Claude chose high/medium/low based on the specific circumstances.

---

## In Practice

### How to interpret confidence:

- **HIGH** → Execute with priority, high success probability
- **MEDIUM** → Execute but monitor for side effects
- **LOW** → Consider as backup if high-confidence plans fail

### How the pipeline uses it:

Currently, confidence is **informational only** — the pipeline generates and executes all plans regardless of confidence. However, you could:

1. **Prioritize by confidence:** Try high-confidence plans first
2. **Filter by confidence:** Skip low-confidence plans if pressed for time
3. **Adjust parameters:** Use more attempts (K) for low-confidence plans

---

## Example Usage in Pipeline

```python
# Sort plans by confidence before execution
plan_priority = {"high": 3, "medium": 2, "low": 1}
sorted_plans = sorted(
    plans,
    key=lambda p: (
        plan_priority.get(p.get("confidence", "low"), 0),
        p.get("evidence", {}).get("ncu_rule_speedup_pct", 0)
    ),
    reverse=True
)

# Execute high-confidence plans first
for plan in sorted_plans:
    if plan["confidence"] == "high":
        execute(plan)
```

This would execute plans in priority order based on confidence and expected speedup.
