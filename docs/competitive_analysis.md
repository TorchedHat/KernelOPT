# kernelopt Competitive Analysis: How We Differ From Every Tool

## Tools We Can Benchmark Against

### In our workspace (have source code):
1. **KernelAgent** (Meta) -- multi-agent LLM kernel generation + optimization
2. **AutoKernel** -- 5-stage correctness + Amdahl's law scheduling
3. **KernelSkill / KernelMem** -- dual memory (long-term YAML + short-term trajectory)
4. **CUDAAgent** (ByteDance) -- CUDA-focused optimization agent
5. **HyperAgents** -- multi-agent coordination framework
6. **AccelOpt** -- GPU kernel acceleration optimizer

### From recent papers (no source, can compare on benchmarks):
7. **KernelBlaster** -- Memory-Augmented In-context RL (MAIC-RL)
8. **EvoKernel** -- value-driven memory for cold-start synthesis
9. **AdaExplore** -- failure-driven adaptation + diversity-preserving search
10. **STARK** -- strategic team of agents for refining kernels

---

## Detailed Comparison: kernelopt vs Each Tool

### 1. kernelopt vs KernelAgent (Meta)

| Aspect | KernelAgent | kernelopt | Why ours is different |
|--------|------------|-----------|---------------------|
| **Synthesis** | LLM writes Triton from scratch (40+ LLM calls, anti-cheating regexes) | torch.compile generates Triton deterministically (0 LLM calls) | Compiler-first: guaranteed correct starting point, no wasted LLM calls on synthesis |
| **Input** | KernelBench PyTorch only | Any: PyTorch, Triton, CUDA, Helion (auto-detected) | Framework-agnostic: one tool for all GPU kernel types |
| **Correctness ref** | LLM generates both kernel AND test (circular) | Original PyTorch is ground truth (for synthesized files) | Independent reference: can't have matching bugs |
| **Multi-kernel** | Ignores -- collapses to first Triton kernel | Profiles ALL kernels, bottleneck discovery, strategy-aware | Full computation visibility |
| **Profiling** | NCU on first Triton kernel only | NCU on ALL kernels, classify user/library/runtime | Honest bottleneck identification |
| **Planning** | Single-phase: LLM reads NCU metrics, proposes plan | Two-phase: Strategy Analysis (macro) -> Plan Generation (micro) | Strategy-level reasoning before detail-level planning |
| **Memory** | Static RAG database (7 code samples, hand-curated) | Dynamic memory bank (grows from every run) + Triton tutorial references | Self-improving: learns from successes AND failures |
| **Search** | Beam search (top-N kernels x M bottlenecks) | N parallel plans per iteration, diversity via NCU rules + strategy | Strategy-guided diversity |
| **Autotuning** | LLM embeds @triton.autotune | Decoupled autotuner (Feature 1, mechanical grid search) | LLM picks algorithm, autotuner picks hardware params |
| **Architecture** | Custom multiprocessing | LangGraph with checkpoint/resume | Recoverable, inspectable pipeline |

**Novelty over KernelAgent**: Compiler-first synthesis + two-phase planning + dynamic memory bank + multi-kernel awareness.

---

### 2. kernelopt vs AutoKernel

| Aspect | AutoKernel | kernelopt |
|--------|-----------|-----------|
| **Starting point** | Hand-written starter kernels per op type (9 Triton, 9 CUDA) | torch.compile generates starters automatically for any op |
| **Correctness** | 5-stage correctness pipeline (shape sweep, dtype-aware tolerance) | LLM-generated reference + atol floor + original PyTorch for synthesized |
| **Op coverage** | 9 predefined types (matmul, softmax, layernorm, etc.) | Any operation torch.compile can handle (hundreds of ops) |
| **Scheduling** | Amdahl's law -- focuses effort on bottleneck stage | Two-phase planning -- strategy decides macro approach |
| **Memory** | None | Dynamic memory bank + reference kernels |

**Novelty over AutoKernel**: No predefined op types -- compiler handles ANY op. Dynamic memory vs none.

---

### 3. kernelopt vs KernelSkill / KernelMem

| Aspect | KernelSkill | kernelopt |
|--------|------------|-----------|
| **Long-term memory** | YAML decision table (bottleneck -> strategy mapping, hand-curated) | JSON memory bank (grows from real runs, not hand-curated) |
| **Short-term memory** | Per-kernel trajectory preventing backtracking | Experience items from summarizer (existing), plus memory bank failures |
| **Memory retrieval** | Rule-based matching on bottleneck category | Structural matching (kernel_type + hardware + outcome) |
| **Synthesis** | LLM generates CUDA from scratch | torch.compile generates Triton (compiler, not LLM) |
| **Results** | 5.44x on KB L1, 2.82x on L2, 1.92x on L3 | TBD (benchmark pending) |

**Novelty over KernelSkill**: Memory grows organically (not hand-curated). Compiler-first synthesis (not LLM CUDA generation). Reference kernels from Triton tutorials (not static code samples).

---

### 4. kernelopt vs KernelBlaster

| Aspect | KernelBlaster | kernelopt |
|--------|--------------|-----------|
| **Learning** | MAIC-RL with persistent CUDA knowledge base | Memory bank with structural matching |
| **Synthesis** | LLM generates CUDA | torch.compile generates Triton |
| **Memory** | Embedding-based retrieval | Structural matching (faster, no embedding model needed) |
| **Results** | 1.43x L1, 2.50x L2, 1.50x L3 | TBD |

**Novelty over KernelBlaster**: No embedding model dependency (simpler, faster retrieval). Triton-native (vs CUDA).

---

### 5. kernelopt vs AdaExplore

| Aspect | AdaExplore | kernelopt |
|--------|-----------|-----------|
| **Failure handling** | Converts failures into reusable validity rules | Memory bank stores failures with constraints |
| **Search** | Diversity-preserving search to escape local optima | N parallel plans with strategy-guided diversity hints |
| **Results** | 3.12x L2, 1.72x L3 | TBD |

**Novelty over AdaExplore**: Compiler-first synthesis (AdaExplore uses LLM synthesis). Two-phase planning vs single-phase.

---

### 6. kernelopt vs STARK

| Aspect | STARK | kernelopt |
|--------|------|-----------|
| **Agents** | Multiple specialized agents (writer, reviewer, debugger) | Planner, executor, correctness, summarizer, strategy analyst |
| **Coordination** | Grounded instruction + dynamic context | LangGraph pipeline with state management |
| **Results** | Up to 16x on individual kernels | 1.57x on kernel_081 activation (TBD on full benchmark) |

---

## What Makes kernelopt Unique (Our Novel Contributions)

### 1. Compiler-First, LLM-Second Architecture
No other tool uses `torch.compile` output as the starting point for LLM optimization.
- KernelAgent: LLM from scratch
- AutoKernel: hand-written starters
- KernelSkill: LLM from scratch (CUDA)
- KernelBlaster: LLM from scratch (CUDA)
- **kernelopt: compiler generates correct Triton, LLM only optimizes**

### 2. Two-Phase Planning with Strategy Analysis
No other tool has a separate strategy analysis node that decides the macro approach before plan generation.
- KernelAgent: single-phase (NCU -> plan)
- AutoKernel: Amdahl's law scheduling (implicit)
- KernelSkill: decision table lookup (rigid)
- **kernelopt: LLM-based strategy (fuse/optimize/rewrite) informed by memory bank + reference kernels**

### 3. Multi-Kernel Bottleneck Discovery
No other tool profiles ALL kernels and classifies them.
- KernelAgent: first Triton kernel only
- Others: assume single kernel per problem
- **kernelopt: profiles ALL launches, classifies user/library/runtime, selects target**

### 4. Framework-Agnostic Input
No other tool auto-detects and handles PyTorch + Triton + CUDA + Helion.
- KernelAgent: KernelBench PyTorch only
- AutoKernel: Triton or CUDA (specified)
- KernelSkill: CUDA only
- **kernelopt: ANY input, auto-detected, unified pipeline**

### 5. Dynamic Memory Bank (Self-Improving)
- KernelAgent: static RAG (7 files, never changes)
- AutoKernel: no memory
- KernelSkill: hand-curated YAML (static)
- **kernelopt: JSON bank grows from every run, ships starter from benchmarks**

### 6. Graceful Degradation
No other tool handles synthesis failure gracefully.
- KernelAgent: fails if LLM can't write Triton
- **kernelopt: synthesis failure -> continues with original file, pipeline runs regardless**

---

## Benchmark Comparison Table (to be filled after TODO 2)

| Tool | KB L1 (100) | KB L2 (100) | KB L3 (50) | KB L4 (20) | SOL (261) |
|------|-------------|-------------|------------|------------|-----------|
| PyTorch eager | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| torch.compile | TBD | TBD | TBD | TBD | TBD |
| **kernelopt** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| KernelAgent | TBD | TBD | TBD | N/A | N/A |
| KernelSkill (paper) | 5.44x | 2.82x | 1.92x | N/A | N/A |
| KernelBlaster (paper) | 1.43x | 2.50x | 1.50x | N/A | N/A |
| AdaExplore (paper) | N/A | 3.12x | 1.72x | N/A | N/A |

Note: Paper numbers are speedup over PyTorch eager. Our numbers will be
measured on H200 (papers used A100/H100). Direct comparison requires
same hardware, but relative ranking is meaningful.

---

## Cross-Tool Benchmarking Plan

### Tools we CAN run (have source):
1. **KernelAgent**: Run on KB L1+L2+L3 (same problems as us)
2. **AutoKernel**: Run on their supported kernel types

### Tools we can COMPARE (paper numbers):
3. **KernelSkill**: 5.44x/2.82x/1.92x on KB L1/L2/L3
4. **KernelBlaster**: 1.43x/2.50x/1.50x on KB L1/L2/L3
5. **AdaExplore**: 3.12x/1.72x on KB L2/L3

### Our benchmark:
- ALL 531 problems (KB L1-L4 + SOL all splits)
- Phase 1: empty memory bank (cold start)
- Phase 2: populated memory bank (warm start)
- Compare both phases to show memory bank value
