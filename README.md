# kernelopt

Multi-agent GPU kernel optimizer. Automatically profiles a kernel with Nsight Compute (NCU), generates diverse optimization plans, applies them, verifies correctness, and iterates — building an experience memory that improves plan quality over time.

Supports **Triton**, **CUDA**, and **Helion** kernels. Powered by **LangGraph** + **Claude on Vertex AI**.

---

## Quick Start

```bash
# 1. Install kernelopt package (REQUIRED FIRST)
cd kernelopt
pip install -e ".[triton]"

# 2. Run automated setup
cd ..
./setup_environment.py

# 3. Load environment
source .env

# 4. Optimize a kernel!
CUDA_VISIBLE_DEVICES=0 kernelopt run tests_suite/fixtures/kernels/matmul_triton.py -T 3 -N 4
```

> **Note:** The package MUST be installed before running `setup_environment.py`. The setup script will verify the installation and guide you through configuring your Google Cloud credentials.

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

## GPU Monitoring

`kernelopt run` automatically displays GPU and system status before profiling and monitors GPU activity during optimization.

### Pre-run Validation

Before starting the optimization pipeline, kernelopt shows:

```
System Status:
  CPU Memory (RAM): 1700 MB available / 1805 MB total (94.2% available)
  GPU Device: GPU 0 (NVIDIA H200)
  GPU Memory (VRAM): 130070 MB free / 143771 MB total (90.5% free)
  GPU Utilization: 0.0%
  GPU Temperature: 30°C
  Active processes: 7 (12920 MB GPU memory)

   Active GPU Processes    
 User      PID  GPU Memory 
 host  2605259     4782 MB 
 host  3331695     3544 MB 
 ...
```

**Features:**
- **System RAM** and **GPU VRAM** clearly labeled to avoid confusion
- **GPU utilization** and **temperature** monitoring
- **Active GPU processes** with memory usage breakdown
- **Warnings** if GPU memory is low (<40% free) or GPU is busy (>20% utilization)

### Per-iteration Monitoring

During optimization, kernelopt monitors GPU activity at the start of each iteration (silent when idle):

```
Iteration 2/5  Best: 1.42x  Memory: 4 items
  GPU activity: GPU utilization: 38.3% | 2 processes using 5248 MB
```

**Non-blocking design:** All monitoring is informational only — warnings are displayed but execution never halts.

### Container Support

When running inside containers (Docker, devcontainers), GPU processes from the host system are labeled as `"host"`:

```
   Active GPU Processes    
 User      PID  GPU Memory 
 host  2605259     4782 MB 
 host  3331695     3544 MB 
  Note: 'host' = process running outside container namespace
```

**Limitations in containers:**
- ✅ GPU memory usage per process (accurate)
- ✅ Total GPU utilization (accurate)
- ✅ System RAM status (accurate)
- ❌ Process usernames (shows "host" for processes outside container)
- ❌ CPU% usage (hidden, not accessible due to PID namespace isolation)

**For full process details**, run kernelopt directly on the host system (not in a container).

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

### Quick Setup (Recommended)

**IMPORTANT:** Install the kernelopt package BEFORE running the setup script.

```bash
# Step 1: Install the package (REQUIRED)
cd kernelopt
pip install -e ".[triton]"   # add triton support

# Step 2: Run interactive environment setup
cd ..
python setup_environment.py
```

The setup script will:
- Check for gcloud CLI (installs if missing)
- Guide you through authentication
- Prompt for your Vertex AI project ID and region
- Create a `.env` file with your configuration
- Verify all dependencies are installed
- Check that kernelopt is properly installed

### Manual Setup

```bash
# 1. Install package (REQUIRED FIRST)
cd kernelopt
pip install -e ".[triton]"   # add triton support
pip install -e ".[dev]"      # add pytest/ruff/mypy
cd ..

# 2. Configure environment
cp .env.example .env
# Edit .env with your values:
#   ANTHROPIC_VERTEX_PROJECT_ID=your-project-id
#   VERTEX_REGION=us-east5

# 3. Set up authentication
gcloud auth application-default login

# 4. Load environment
source .env
```

**System Requirements:**
- Python 3.10+ (**verified by setup script**)
- `ncu` (Nsight Compute) on `$PATH` (**verified by setup script**)
- CUDA-capable GPU with compute capability >= 6.0 (**verified by setup script**)
- Google Cloud project with Vertex AI API enabled (**verified by setup script**)
- gcloud CLI installed (**setup script can install this**)

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
| `-T`, `--iterations` | 5 | Max optimization iterations (omit = default 5; with `--memory` + checkpoint: from checkpoint) |
| `-N`, `--plans` | 4 | Plans per iteration (omit = default 4; with `--memory` + checkpoint: from checkpoint) |
| `-K`, `--attempts` | 1 | Executor attempts per plan (omit = default 1; with `--memory` + checkpoint: from checkpoint) |
| `--top-k` | 4 | Max experience items added per iteration |
| `-o`, `--output-dir` | `pipeline_output/` | Output **base** directory for `kernelopt run` (see Outputs) |
| `--run-subdir` / `--no-run-subdir` | on | **Every** `kernelopt run` creates `run_<timestamp>/` under `-o` (default: on) |
| `--csv` | — | Skip baseline profiling, use existing NCU CSV |
| `--memory` | — | **Resume** from a run directory. If `pipeline_checkpoint.json` exists: **reload full pipeline config from it** (NCU names, `kernel_args`, models, project, etc.); only **`-T` / `-N` / `-K`** may be overridden from the CLI (omit those to use saved values). Other flags are ignored. Continues from the last checkpoint stage. If no checkpoint: baseline CSV + context fallback (then normal CLI applies). Same kernel **basename** required. `-o` / `--run-subdir` ignored. Example Usage: kernelopt run tests_suite/fixtures/kernels/matmul_triton.py -T 3 -N 4 -K 1 --memory pipeline_output/run_20260414_063558_408400|
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

# CUDA kernel — skip baseline (use existing CSV from a prior run folder)
kernelopt run my_kernel.cu --csv pipeline_output/run_20260413_120000_000000/baseline.csv

# Resume (only kernel path + run dir required; T/N/K from checkpoint unless you pass -T -N -K)
kernelopt run softmax.py --memory pipeline_output/run_20260413_120000_000000/

# Resume with more iterations only
kernelopt run softmax.py -T 5 --memory pipeline_output/run_20260413_120000_000000/

# Plan-only: pass experience JSON with -E/--experience (not --memory)
kernelopt plan profile.csv softmax.py -N 2 -E pipeline_output/run_20260413_120000_000000/optimization_memory.json

# Flat layout under -o (no run_* subfolder), same as older behavior
kernelopt run softmax.py -o pipeline_output --no-run-subdir

# Profile only (no optimization)
kernelopt profile softmax.py -o pipeline_output/baseline

# Inspect a profiling CSV
kernelopt show pipeline_output/baseline.csv

# Large input that OOMs with kernel replay
kernelopt run big_kernel.py --replay-mode application
```

---

## Outputs

**`kernelopt run` only:** by default, **every** time you run `kernelopt run`, artifacts go in a **new** timestamped folder under `--output-dir`:

`pipeline_output/run_YYYYMMDD_HHMMSS_microseconds/`

Other commands (`profile`, `show`, `plan`, …) are unchanged and do not use this layout.

Use **`--no-run-subdir`** on `kernelopt run` to write directly into `-o` (flat layout, overwrites on repeat runs).

All artifact paths below are relative to that **run directory**:

| File | Description |
|------|-------------|
| `best_kernel{ext}` | Best optimized kernel source |
| `optimization_memory.json` | Accumulated experience items |
| `pipeline_checkpoint.json` | Updated after each major stage: baseline, **plans**, **execute** (pre-NCU), **summarize**, **select**, then each **iteration** — resume continues from the last completed stage (same plans are not regenerated when resuming after `after_plans`) |
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
│   ├── checkpoint.py            # Resume checkpoints (--memory run dir)
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
│       ├── gpu_monitoring.py    # GPU/system monitoring utilities
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

