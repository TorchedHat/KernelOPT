# Agentic Pipeline Improvements: Research Synthesis & Recommendations

**Date:** 2026-05-24
**Sources:** KernelEvolve (Meta), Reliability Framework (Khanal et al.), Anthropic Harnesses, Anthropic Context Engineering

---

## Step 1: Synthesized Research Findings

### Source 1: KernelEvolve (Meta) — Production-grade agentic kernel optimization

**Key patterns directly applicable to our pipeline:**

1. **Graph-based search over kernel variants.** KernelEvolve models optimization as a search graph G=(V,E) where each node is a kernel variant and edges are transformations. Selection policies (greedy, MCTS, evolutionary) choose which nodes to expand. Our pipeline currently uses linear iteration — each iteration picks the best kernel and optimizes from there. This is pure greedy search.

2. **Two-tier context: Context Memory Sub-Agent + Deep Search Sub-Agent.** The context memory sub-agent maintains a relational metadata store (id, parent_id, score, is_buggy, path_ref) linked to an object store (kernel files, profiling results, analysis reports). The deep search sub-agent retrieves from a persistent knowledge base organized as a hierarchical file system. This separates "what happened" from "what to try next."

3. **Persistent knowledge base as hierarchical filesystem.** Constraints, guidance, and hardware-specific knowledge stored as markdown files in a directory tree with an index.md. The LLM navigates this tree using file/folder names as retrieval signals — no embeddings needed.

4. **Universal operator with retrieval-augmented dynamic prompting.** Instead of fixed prompt templates per optimization type, KernelEvolve synthesizes prompts at runtime from profiling results + knowledge base + search history. Our pipeline's planner does something similar but with less dynamic context injection.

5. **Fitness = speedup with correctness as hard constraint.** F(v) = t_baseline / t_optimized, with F(v)=0 for incorrect kernels. Our pipeline already follows this pattern.

6. **Checkpointing every node.** Every kernel variant is persisted with its profiling results, enabling resume from any point. Our pipeline has checkpointing but it's coarse-grained (per-iteration, not per-candidate).

### Source 2: Reliability Framework (Khanal et al.) — Formal metrics for agent reliability

**Key patterns:**

7. **Meltdown detection via tool-call entropy.** Track sliding-window entropy over agent actions. If the executor keeps making semantically identical rewrites, that's a meltdown signal. Intervention: checkpoint best kernel, restart planner with fresh context.

8. **Task decomposition is the highest-leverage reliability intervention.** Reliability decays super-linearly with task duration. Each optimization iteration should be a self-contained short-horizon task, not a monolithic session. Our pipeline already does this — the profiler→planner→executor→verifier loop is well-decomposed.

9. **Naive memory scaffolds hurt at long horizons.** Growing episodic memory consumes context and step budget. Don't accumulate raw profiling history. Use compressed, structured summaries. Expire old data once summarized.

10. **Context reset on meltdown, not abandonment.** Save state, clear context, re-enter from profiler with only the compressed summary. Don't abort the entire optimization.

11. **Partial credit matters.** Track partial progress: "improved memory throughput but broke correctness" is useful signal for the planner.

### Source 3: Anthropic Harnesses — Long-running agent patterns

**Key patterns:**

12. **Structured JSON task lists over markdown.** JSON prevents inappropriate model edits. Each task has status, steps, and verification criteria. Our optimization_memory uses JSON — good.

13. **Session startup ritual.** Every iteration starts by reading: progress file, git log, current state. Eliminates "guess what happened" failures. Our pipeline should formalize this — the planner should always receive a structured summary of all prior iterations.

14. **One optimization per session.** Don't batch. Each iteration: one plan, one rewrite, one verification. Our pipeline applies N plans per iteration — this is a divergence from the recommended pattern. Consider whether single-plan iterations with more iterations is better.

15. **Clean state between sessions.** Commit after every meaningful unit. Leave the codebase mergeable. Our pipeline checkpoints after each stage — good.

16. **Testing is first-class, never assumed.** Models skip verification unless forced. Our correctness gate is mandatory — good.

### Source 4: Anthropic Context Engineering — Token-efficient agent design

**Key patterns:**

17. **Tool result clearing.** Once a tool executes deep in history, agents rarely need raw results. Clear old tool outputs to save context. Our pipeline doesn't do this — profiling CSVs and NCU outputs accumulate.

18. **Just-in-time context retrieval.** Don't preload all data. Maintain lightweight identifiers and load on demand. Our planner receives the full profiling context upfront — consider whether a summary + on-demand detail retrieval would be more efficient.

19. **Minimal viable prompt + iterative refinement.** Start with the smallest high-signal prompt. Add instructions based on observed failures, not preemptively. Our executor prompt is already fairly lean.

20. **Multi-agent with clean context windows.** Sub-agents for focused tasks, returning condensed summaries. Our pipeline uses separate agents (profiler, planner, executor, summarizer) — this is the right pattern.

---

## Step 2: Gap Analysis Against Our Pipeline

| # | Technique | KernelOPT Status | Gap |
|---|-----------|-----------------|-----|
| 1 | Graph-based search | Linear greedy only | **Major** — no tree/MCTS exploration |
| 2 | Two-tier context (metadata + objects) | In-memory state dict | **Major** — no persistent metadata store |
| 3 | Persistent knowledge base | optimization_memory.json (flat list) | **Medium** — no hierarchical organization |
| 4 | Dynamic prompt synthesis from KB | Static planner prompt | **Medium** — fixed prompt structure |
| 7 | Meltdown detection | None | **Medium** — no detection of stuck loops |
| 9 | Context compaction for memory | Raw profiling data accumulates | **Medium** — summarizer exists but old data not expired |
| 10 | Context reset on failure | Retry with error feedback | **Low** — already implemented |
| 11 | Partial credit tracking | Binary pass/fail only | **Medium** — useful signal lost |
| 14 | One plan per iteration | N plans per iteration | **Architectural** — different tradeoff |
| 17 | Tool result clearing | Not implemented | **Low** — LangGraph manages context |

---

## Step 3: Prioritized Recommendations

### P0: Tree Search Over Kernel Variants (from KernelEvolve)

**What:** Replace the current linear `best_kernel → optimize → new best` loop with a search tree. Each candidate becomes a node. The planner selects which node(s) to expand based on fitness scores and diversity.

**Why:** Our pipeline currently discards all non-best candidates after each iteration. A candidate that's 0.95x might lead to 1.5x after a second optimization — but we'll never know because we threw it away.

**How:**
1. Add a `SearchTree` class that persists nodes (kernel source, fitness, profiling context, parent_id)
2. Replace `select_best_candidate` with a selection policy: greedy (current behavior), or UCT-based (explore underexplored branches)
3. Allow the planner to propose optimizations on any node, not just the current best
4. Store tree as JSON in the output directory for visualization and resume

**Effort:** Large. This changes the core iteration loop.

### P1: Persistent Metadata Store (from KernelEvolve)

**What:** Replace the in-memory `GraphState` accumulation with a two-tier store: lightweight metadata (SQLite or JSON) + object store (kernel files on disk).

**Why:** As iterations grow, the state dict bloats. The LangGraph state carries full kernel sources and profiling contexts across all iterations. A metadata store enables efficient querying ("what was the best kernel at iteration 3?") without loading everything into context.

**How:**
1. Each candidate gets a row: `(id, parent_id, iteration, plan_direction, fitness, correctness, latency_ns, kernel_path, profiling_path)`
2. Replace `iteration_results` list with metadata store queries
3. The planner receives a structured summary from the store, not raw state

**Effort:** Medium. Incremental — can coexist with current state.

### P2: Meltdown Detection (from Reliability Framework)

**What:** Track optimization diversity across iterations. If the executor produces semantically identical rewrites 3+ times, trigger a context reset.

**Why:** We've seen the pipeline produce the same "increase XBLOCK and num_warps" optimization repeatedly. This wastes API calls and profiling time.

**How:**
1. Hash the `direction` + `change_summary` from each plan
2. If 3 consecutive plans have similar hashes, inject a "diversity boost" into the planner prompt: "The following approaches have been tried and showed diminishing returns: [...]. Try a fundamentally different strategy."
3. If 5 consecutive plans repeat, save best kernel, clear optimization memory, restart with fresh context

**Effort:** Small. Can be added to `generate_plans()`.

### P3: Hierarchical Knowledge Base (from KernelEvolve)

**What:** Organize the optimization memory as a hierarchical file system instead of a flat JSON list.

**Why:** Our current `optimization_memory.json` is a flat list of dicts. As it grows (10+ items), the planner receives a wall of text. KernelEvolve's approach — organized by hardware, optimization type, kernel pattern — enables targeted retrieval.

**How:**
1. Directory structure: `memory_bank/{hardware}/{kernel_type}/{optimization_pattern}.json`
2. Index file listing available memories with one-line summaries
3. Planner receives the index, retrieves specific entries on demand
4. Already partially implemented via `memory_bank.py` — extend with hierarchy

**Effort:** Medium.

### P4: Structured Iteration Log (from Anthropic Harnesses)

**What:** Write a structured JSON log after each iteration capturing: what was tried, what worked, what failed, why. The planner reads this at the start of each iteration.

**Why:** Currently the planner receives the profiling context and optimization memory, but doesn't get a clear "here's what happened in iterations 1-3" briefing. This causes redundant strategies.

**How:**
1. After `select_best`, write `iteration_log.json` with: `[{iteration, plans_tried: [{direction, speedup, correctness, key_insight}], best_so_far, remaining_opportunities}]`
2. Planner prompt includes this log as "Previous Iterations" section
3. This replaces the current approach of passing raw `correctness_failure_items`

**Effort:** Small. Mostly prompt engineering.

### P5: Partial Credit in Correctness (from Reliability Framework)

**What:** When correctness fails, still record which metrics improved. Feed this back to the planner.

**Why:** A rewrite that improves memory bandwidth by 30% but introduces a numerical bug contains valuable signal. The planner should know "this direction was promising but broke correctness at high values."

**How:**
1. In `execute_plans`, when correctness fails, still profile the kernel (if it runs)
2. Record partial metrics: `{direction, correctness: "failed", latency_ns: 3200, improvement_pct: 40, failure_reason: "max_diff=1.2e-3"}`
3. Planner prompt includes partial results under "Promising but incorrect approaches"

**Effort:** Small. Extends existing `correctness_failure_items`.

### P6: Context Compaction for Multi-Iteration Runs (from Anthropic Context Engineering)

**What:** After each iteration, compact the profiling context to only the metrics that changed. Expire raw NCU sections after the summarizer extracts insights.

**Why:** By iteration 3, the context carries 3 full profiling contexts (each ~2000 tokens). The planner only needs the current context + a summary of what changed.

**How:**
1. After summarizer runs, replace the previous profiling context with a compressed diff: "Iter 1→2: L1 hit rate improved 45%→72%, occupancy unchanged"
2. Only keep the current iteration's full context
3. Already partially done by the summarizer — formalize the compaction

**Effort:** Small.
