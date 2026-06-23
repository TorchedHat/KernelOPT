# KernelOPT Roadmap

**Last updated:** 2026-05-24
**Branch:** KernelOPT

---

## Current State

KernelOPT is a multi-agent GPU kernel optimization pipeline. Given a PyTorch model or a standalone Triton/CUDA/Helion kernel, it:

1. **Synthesizes** — torch.compile → extract Triton kernels from Inductor output
2. **Profiles** — NCU sections-based CSV → structured profiling context
3. **Plans** — LLM generates N optimization plans from profiling evidence
4. **Executes** — LLM rewrites the kernel body per plan
5. **Verifies** — kernel-level correctness (baseline vs optimized) + model-level correctness (re-stitched into full compiled model)
6. **Profiles candidates** — NCU on each correct candidate
7. **Summarizes** — extracts optimization insight for memory bank
8. **Selects** — picks best candidate, loops for T iterations

### What works today
- Full pipeline runs end-to-end for PyTorch, Triton, CUDA, Helion inputs
- Kernel-level correctness gate: compares baseline standalone vs optimized standalone (20/20 across test runs)
- Re-stitching: swaps optimized @triton.jit body back into async_compile.triton() block in synthesized_triton.py
- Triton-only targeting: skips cuBLAS/cuDNN/runtime kernels for synthesized models
- In-place kernel aliasing fix: clones inputs before comparison
- Optimization memory: cross-run learning from successes and failures
- Checkpointing: resume from any iteration

### What's broken / incomplete
- **Model-level correctness comparison** — re-stitched file runs but comparison shows max_diff=5.9e4. Root cause: unclear whether the kernel body replacement is producing a semantically different kernel, or the comparison itself is flawed (different random weights across module loads). Needs debugging.
- **End-to-end model timing** — not yet implemented. Pipeline reports kernel-level speedup only.
- **Speedup numbers are misleading** for multi-kernel models — e.g., 594x "speedup" because the baseline latency came from a different kernel than the one being optimized. Fixed by triton-only targeting but needs validation.

---

## Research Findings (2026-05-24)

### Source 1: KernelEvolve — Meta (arXiv 2512.23236v3)

Production-grade agentic kernel optimization deployed at Meta. 100% pass rate on KernelBench (250 problems), 1.25-17x speedups, kernels for NVIDIA/AMD/MTIA.

**Key techniques we should adopt:**

| Technique | What it does | Our gap |
|-----------|-------------|---------|
| **Graph-based search (MCTS, evolutionary)** | Models optimization as a tree. Nodes = kernel variants, edges = transformations. Selection policy (greedy/UCT/evolutionary) picks which nodes to expand. Enables backtracking and branching. | We do linear greedy — always optimize from the single best kernel. Discards promising non-best candidates. |
| **Two-tier context memory** | Relational metadata store (id, parent_id, score, correctness, path_ref) + object store (kernel files, profiling). Enables efficient querying across hundreds of candidates. | We use in-memory GraphState dicts. State bloats across iterations. |
| **Persistent knowledge base as filesystem** | Hierarchical directory: constraints/, guidance/, hardware/. Each with 15-40 markdown docs per platform. LLM navigates via file/folder names. index.md as table of contents. | Our optimization_memory is a flat JSON list. No hierarchical organization, no hardware-specific guidance. |
| **Deep search sub-agent** | Retrieves relevant optimization examples from the knowledge base based on the current profiling bottleneck. Injects "here's how a similar kernel was optimized" into the prompt. | We pass optimization_memory items but don't retrieve by similarity. No curated reference kernel library. |
| **Context memory sub-agent** | Analyzes search tree state to guide exploration. Identifies which optimization challenges need attention. Conditions knowledge retrieval on runtime analysis. | Our planner receives raw profiling context without analysis of what's been tried. |
| **Universal operator (dynamic prompting)** | Single agentic kernel generator with retrieval-augmented prompts. No fixed templates — prompt synthesized at runtime from profiling + knowledge base + search history. | Our executor prompt is mostly static. Plan + kernel + profiling summary, but no dynamic example injection. |
| **Multi-pass profiling (MPP)** | Federated profiler: NCU + Proton (instruction-level) + system profiler. Unified cross-stack metrics. | We use NCU sections only. No instruction-level profiling. |
| **Evaluation on FaaS** | Kernel evaluation runs on Function-as-a-Service infrastructure for isolation and scalability. | We run on the same GPU. No isolation. |

### Source 2: Reliability Framework — Khanal et al. (arXiv 2603.29231v1)

Formal framework for evaluating agent reliability over long-horizon tasks. 23,392 episodes, 10 models.

**Key techniques:**

| Technique | What it does | Our gap |
|-----------|-------------|---------|
| **Meltdown detection (MOP)** | Sliding-window entropy over tool-call sequences. Detects when agent enters incoherent looping. Intervention: checkpoint + restart with fresh context. | No detection of stuck loops. Pipeline can waste iterations producing identical optimizations. |
| **Task decomposition** | Reliability decays super-linearly with duration. Short, independently verifiable sub-tasks dramatically improve success probability. | Our pipeline is already well-decomposed (profile→plan→execute→verify). Validated by this research. |
| **Naive memory hurts** | Episodic scratchpad memory either hurts or is neutral for long tasks. Growing memory consumes context budget. | We accumulate optimization_memory items indefinitely. Should expire/compact old entries. |
| **Context reset on meltdown** | Don't abort — save state, clear context, re-enter from profiler with compressed summary. | We retry with error feedback but don't do full context resets. |
| **Partial credit (GDS)** | Binary pass/fail obscures progress. Track partial metrics even for incorrect kernels. | We discard all information from failed correctness checks except the error message. |
| **Error correlation** | Once an agent forms a wrong hypothesis, it persists. Breaking error chains (via verification) matters more than reducing per-step error rate. | Our verifier after each iteration serves this purpose. Validated by this research. |

### Source 3: Anthropic — Effective Harnesses for Long-Running Agents

Patterns for building harnesses that maintain agent coherence across context windows.

**Key techniques:**

| Technique | What it does | Our gap |
|-----------|-------------|---------|
| **Two-agent architecture** | Initializer agent expands goal into JSON task list. Worker agent picks one task per session. | We could separate "strategy analysis" (initializer) from per-iteration execution (worker). Already partially done with analyze_strategy. |
| **Session startup ritual** | Fixed sequence: read progress, review git log, select task, smoke test, then work. Eliminates context reconstruction waste. | Our planner doesn't get a structured "here's what happened in previous iterations" briefing. |
| **JSON over markdown for task state** | JSON prevents inappropriate model edits. Structured, parseable, less prone to corruption. | Our optimization_memory is JSON. Good. |
| **One task per session** | Prevents context exhaustion and half-finished work. | We apply N plans per iteration. Different tradeoff — N plans enables parallelism but may cause context bloat in the planner. |
| **Clean state between sessions** | Commit after every unit. Leave codebase mergeable. | Our checkpointing does this. Good. |
| **Testing is first-class** | Models skip verification unless forced. Must be mandatory and automated. | Our correctness gate is mandatory. Good. |

### Source 4: Anthropic — Effective Context Engineering for AI Agents

Token-efficient patterns for managing agent context.

**Key techniques:**

| Technique | What it does | Our gap |
|-----------|-------------|---------|
| **Tool result clearing** | Old tool outputs rarely needed. Clear deep-history results to save context. | Profiling CSVs and NCU outputs accumulate in our pipeline state. |
| **Just-in-time retrieval** | Don't preload all data. Maintain identifiers, load on demand via tools. | Our planner receives full profiling context upfront. Could use summary + on-demand detail. |
| **Minimal viable prompt** | Start small, add instructions based on observed failures. Don't preemptively stuff edge cases. | Our executor prompt is reasonably lean. Good. |
| **Multi-agent with clean contexts** | Sub-agents for focused tasks, return condensed summaries (1-2k tokens). | Our pipeline uses separate agents. Good. |
| **Structured note-taking** | Agent writes persisted notes outside context window. Retrieves later. | Our summarizer does this. The optimization_memory is the persistent notes. |

---

## Build Plan

### Phase 1: Fix what's broken

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Debug model-level correctness (max_diff=5.9e4) | TODO | Re-stitched file runs but outputs differ. Need to verify the kernel body replacement is semantically correct. |
| 1.2 | Validate full pipeline with working model correctness | TODO | Run L1 + L3 kernels, all gates pass. |

### Phase 2: Improve code generation quality

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Reference kernel library | TODO | Curated before/after pairs by pattern (pointwise, reduction, matmul, fused) and hardware. Wire into executor prompt. Highest-leverage improvement for code quality. |
| 2.2 | Structured iteration log | TODO | JSON log after each iteration: plans tried, speedups, failures, key insights. Planner reads this to avoid redundant strategies. Small effort, immediate payoff. |
| 2.3 | Meltdown detection | TODO | Hash plan directions, inject diversity prompt after 3 repeats. Checkpoint + context reset after 5 repeats. Prevents wasted iterations. |
| 2.4 | Partial credit tracking | TODO | Record metrics from incorrect kernels. Feed back to planner as "promising but incorrect approaches." |

### Phase 3: Better search

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Tree search over candidates | TODO | SearchTree class: nodes = kernel variants, edges = transformations. Selection policies: greedy (current), UCT (explore-exploit), evolutionary (crossover). Big architectural change. |
| 3.2 | Persistent metadata store | TODO | SQLite or JSON. Each candidate: (id, parent_id, iteration, direction, fitness, correctness, latency_ns, kernel_path, profiling_path). Replaces in-memory iteration_results. |
| 3.3 | Hierarchical knowledge base | TODO | Directory structure: memory_bank/{hardware}/{kernel_type}/{pattern}.json. Index file. On-demand retrieval. Replaces flat optimization_memory list. |

### Phase 4: Scale

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Parallel sub-agents for top-K kernels | TODO | Optimize multiple Triton kernels simultaneously for multi-kernel models. Requires solid re-stitching. |
| 4.2 | End-to-end model timing | TODO | Wall-clock comparison of full model with re-stitched kernel vs original. |
| 4.3 | Context compaction | TODO | Compress old profiling contexts to diffs after summarizer runs. Expire raw NCU data. |

---

## References

- KernelEvolve: `misc/2512.23236v3.pdf` — Meta's production kernel optimization system
- Reliability Framework: `misc/2603.29231v1.pdf` — Formal metrics for long-horizon agent reliability
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Design spec: `docs/superpowers/specs/2026-05-24-agentic-pipeline-improvements.md`
- Re-stitching spec: `docs/superpowers/specs/2026-05-21-kernel-restitching-design.md`
