# kernelopt

Multi-agent GPU kernel optimizer. Automatically profiles a kernel with Nsight Compute (NCU), generates diverse optimization plans, applies them, verifies correctness, and iterates — building an experience memory that improves plan quality over time.

Supports **Triton**, **CUDA**, and **Helion** kernels. Powered by **LangGraph** + **Claude on Vertex AI**.

---

## How it works

Each iteration runs six phases in a LangGraph pipeline:

```
profile_baseline
       │
       ▼
generate_plans (N plans, concurrent)
       │
       ▼
execute_plans (executor agent × N, correctness gate)
       │
       ▼
profile_candidates (NCU on passing candidates)
       │
       ▼
summarize_results (update optimization memory)
       │
       ▼
select_best ──► increment_iteration ──► (loop or stop)
```

**Stopping conditions**: max T iterations reached, 2 consecutive iterations with no improvement, or all plans report low confidence.

---

## Agents

### Planner (`agents/planner/`)

LangGraph sub-graph: `agent → tools → agent → ... → submit_plan`.

- N instances run **concurrently** via `ThreadPoolExecutor`.
- Each instance receives a diversity hint pointing it at a different NCU optimization rule.
- Cross-iteration direction history prevents repeating tried strategies.
- Output: an `OptimizationPlan` (Pydantic-validated) with `direction`, `evidence`, `steps`, and `confidence`.

### Executor (`agents/executor/`)

LangGraph sub-graph: `agent → tools → agent → ... → submit_kernel`.

- Reads the current best kernel and the planner's optimization plan.
- Up to K attempts; each `submit_kernel` call validates kernel markers (`@triton.jit`, `__global__`, `@helion.kernel`, `@hl.kernel`).
- Returns `(optimized_source, execution_result)` with `ready_for_profiling` flag.

### Profiler (`agents/profiler.py`)

Stateless — no LLM involved.

1. Runs `ncu --set full` → `.ncu-rep`
2. Exports CSV via `ncu --import`
3. Parses sections, rules, and launch stats into a structured `profiling_context` dict
4. Auto-detects framework via AST (Python) or regex (CUDA)
5. Enriches context with GPU hardware specs (SM count, HBM bandwidth, compute capability)

### Summarizer (`agents/summarizer.py`)

Single LLM call with structured output.

- Diffs the slow/fast kernel pair.
- Produces an `experience_item` (generalizable insight) and a `memory_update` action (`append` / `replace` / `skip`).
- Correctness-failure items (no profiling) bypass speedup thresholds and go directly into memory so the planner avoids repeating broken implementations.

### Correctness Checker (`functions/correctness.py`)

- **Agentic mode** (default): Claude generates a CPU reference implementation and validates numerical outputs.
- **Non-agentic mode**: `--no-agentic-correctness` flag uses a simpler static check.

---

## Optimization Memory

A bounded FIFO queue (capacity `ExpN`, default 8) of `experience_items`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ExpN` | 8 | Queue capacity |
| `TopK` | 4 | Max items added per iteration |
| `t_pos` | 1.04 | Min speedup to store a positive item |
| `t_neg` | 1.15 | Min slowdown factor to store a negative item |

Each item records: `direction`, `speedup`, `rewrite_type` (positive/negative), `profiling_signal`, `strategy_description`, `slow_pseudocode`, `fast_pseudocode`, `applicable_when`, `do_not_apply_when`.

The planner reads this memory to avoid dead ends and reinforce strategies that worked.

---

## State

| Class | Role |
|-------|------|
| `PipelineConfig` | User-facing parameters (T, N, K, TopK, model, NCU flags, …) |
| `PipelineState` | Full mutable runtime state (best kernel, memory, iteration results) |
| `GraphState` | LangGraph `TypedDict` projection of `PipelineState` |
| `PlanCandidate` | One plan's execution + profiling results |
| `IterationResult` | All candidates for one iteration + which was kept |

---

## Installation

```bash
cd kernelopt
pip install -e ".[triton]"   # add triton support
pip install -e ".[dev]"      # add pytest/ruff/mypy
```

Requires `ncu` (Nsight Compute) on `$PATH` and a GPU.

---

## CLI

```
kernelopt run <kernel.py|.cu|.hl>   # full optimization pipeline
kernelopt profile <kernel>          # single NCU profiling run
kernelopt show <ncu.csv>            # pretty-print a profile CSV
kernelopt plan <ncu.csv> <kernel>   # generate plans only (no execution)
kernelopt kernels <ncu.csv>         # list all kernels in an NCU CSV
```

### `kernelopt run` options

| Flag | Default | Description |
|------|---------|-------------|
| `-T`, `--iterations` | 5 | Max optimization iterations |
| `-N`, `--plans` | 4 | Plans generated per iteration |
| `-K`, `--attempts` | 1 | Executor attempts per plan |
| `--top-k` | 4 | Max experience items added per iteration |
| `-o`, `--output-dir` | `pipeline_output/` | Output directory |
| `--csv` | — | Skip baseline profiling, use existing NCU CSV |
| `--memory` | — | Load an existing `optimization_memory.json` |
| `--kernel-name` | auto | NCU `--kernel-name` regex |
| `--launch-skip` | 0 | NCU `--launch-skip` |
| `--launch-count` | — | NCU `--launch-count` |
| `--replay-mode` | `kernel` | `kernel` or `application` |
| `--kernel-args` | — | Extra args forwarded to the kernel program |
| `--model` | `claude-sonnet-4@20250514` | Default model for all agents |
| `--planner-model` | — | Model override for the planner |
| `--executor-model` | — | Model override for the executor |
| `--summarizer-model` | — | Model override for the summarizer |
| `--project-id` | `$ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project for Vertex AI |
| `--region` | `us-east5` | Vertex AI region |
| `--agentic-correctness / --no-agentic-correctness` | on | Use Claude for correctness checks |

### Examples

```bash
# Triton softmax — 3 iterations, 4 plans, profile only kernels named 'softmax_kernel'
kernelopt run softmax.py -T 3 -N 4 --kernel-name 'softmax_kernel' --kernel-args '--profile-only'

# CUDA kernel — skip baseline (use existing CSV)
kernelopt run my_kernel.cu --csv pipeline_output/baseline.csv

# Resume from saved memory
kernelopt run softmax.py --memory pipeline_output/optimization_memory.json

# Profile only (no optimization)
kernelopt profile softmax.py -o pipeline_output/baseline

# Inspect a profiling CSV
kernelopt show pipeline_output/baseline.csv

# Large input that OOMs with kernel replay
kernelopt run big_kernel.py --replay-mode application
```

---

## Outputs

All files land in `--output-dir` (default `pipeline_output/`):

| File | Description |
|------|-------------|
| `best_kernel{ext}` | Best optimized kernel source |
| `optimization_memory.json` | Accumulated experience items |
| `optimization_log.md` | Full per-iteration log with metric comparison tables |
| `baseline.csv` / `baseline.ncu-rep` | Baseline profiling artifacts |
| `{kernel}_iter{N}_plan{L}_{direction}.{ext}` | Each candidate kernel |
| `{stem}.csv` / `{stem}.ncu-rep` / `{stem}_profiling_context.json` | Per-candidate profiling artifacts |

---

## Project layout

```
kernelopt/
├── kernelopt/
│   ├── cli.py                   # Typer CLI entry point
│   ├── graph.py                 # Imperative pipeline (reference impl)
│   ├── langgraph_pipeline.py    # LangGraph pipeline (active impl)
│   ├── state.py                 # PipelineConfig, PipelineState, GraphState, PlanCandidate, IterationResult
│   ├── memory.py                # Bounded experience queue
│   ├── llm.py                   # LLM builder (Anthropic / Vertex AI)
│   ├── schemas.py               # Shared Pydantic schemas
│   ├── agents/
│   │   ├── planner/
│   │   │   ├── graph.py         # Planner LangGraph sub-graph
│   │   │   ├── prompts.py       # System prompt + diversity hints
│   │   │   └── tools.py         # PlannerToolServer + submit_plan
│   │   ├── executor/
│   │   │   ├── graph.py         # Executor LangGraph sub-graph
│   │   │   ├── prompts.py       # System prompt + initial message builder
│   │   │   └── tools.py         # ExecutorToolServer + submit_kernel
│   │   ├── profiler.py          # NCU runner + profiling_context builder
│   │   ├── summarizer.py        # Summarizer LLM call (structured output)
│   │   ├── correctness.py       # Agentic correctness agent
│   │   ├── gpu_specs.py         # GPU hardware spec lookup
│   │   └── schemas.py           # Agent-level Pydantic schemas (OptimizationPlan, SummarizerOutput)
│   └── functions/
│       ├── csv_parser.py        # NCU CSV parser
│       ├── correctness.py       # Correctness check dispatcher
│       └── selector.py          # Candidate selector utilities
└── tests_suite/              # Unified test suite (366 tests)
    ├── unit/                 # Unit tests (no GPU required)
    ├── component/            # Component tests with mocks
    │   └── agents/
    │       ├── test_planner_graph.py
    │       ├── test_planner_tools.py
    │       ├── test_executor_graph.py
    │       └── test_executor_tools.py
    ├── integration/          # Integration tests
    ├── e2e/                  # End-to-end pipeline tests
    ├── quality/              # Architecture & API stability tests
    ├── fixtures/             # Test kernels and sample data
    └── README.md             # Complete testing guide
```

> **Testing**: See `tests_suite/README.md` for the complete testing guide. Run tests with `pytest tests_suite/ --gpu <id>` or use interactive GPU selection with `pytest tests_suite/`.

---

