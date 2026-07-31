"""Data model for compiler-extracted context.

All dataclasses are organized by compiler layer:
  - Dynamo: pre-decomposition graph structure + graph breaks
  - AOTAutograd: decomposition path records
  - Inductor: IR nodes, fusion decisions, autotune state, codegen strategy
  - Triton: TTIR/TTGIR text, compilation metadata, register pressure
  - XLA: StableHLO, optimized HLO, fusion dumps, autotune results

The top-level CompilerContext holds all layers as optional fields.
All fields populated during synthesis; immutable thereafter.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional


# =========================================================================
# Dynamo layer
# =========================================================================

@dataclass(frozen=True)
class DynamoOpInfo:
    """Single op from the pre-decomposition FX graph."""

    op_name: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    is_compound: bool


@dataclass(frozen=True)
class GraphBreakInfo:
    """A single Dynamo graph break."""

    reason: str
    location: str
    ops_before: int
    ops_after: int


@dataclass(frozen=True)
class DynamoGraphInfo:
    """Pre-decomposition FX graph structure from TorchDynamo."""

    original_ops: tuple[DynamoOpInfo, ...] = ()
    graph_breaks: tuple[GraphBreakInfo, ...] = ()
    num_subgraphs: int = 1
    unsupported_ops: tuple[str, ...] = ()


# =========================================================================
# AOTAutograd layer
# =========================================================================

@dataclass(frozen=True)
class AOTDecompInfo:
    """Records how AOTAutograd decomposed a single compound op."""

    original_op: str
    decomposed_into: tuple[str, ...]
    num_intermediates: int
    decomposition_source: str = ""


# =========================================================================
# Inductor layer
# =========================================================================

@dataclass(frozen=True)
class ReductionInfo:
    """Metadata about a single reduction from Inductor IR."""

    reduction_type: str
    reduction_ranges: tuple[int, ...]
    reduction_hint: str
    src_dtype: str
    dst_dtype: str
    is_persistent: bool = False
    is_cooperative: bool = False
    rnumel_padded: int = 0
    wasted_sram_fraction: float = 0.0


@dataclass(frozen=True)
class IRNodeInfo:
    """Compiler-extracted metadata for a single Inductor buffer."""

    buffer_name: str
    node_type: str
    iteration_ranges: tuple[int, ...]
    total_elements: int
    dtype: str
    bytes_per_element: int
    total_memory_bytes: int
    num_reads: int = 0
    num_writes: int = 0
    read_buffer_names: tuple[str, ...] = ()
    write_buffer_names: tuple[str, ...] = ()
    reduction: Optional[ReductionInfo] = None


@dataclass(frozen=True)
class FusionRejection:
    """A fusion that Inductor considered and rejected."""

    node1_name: str
    node2_name: str
    reason: str
    near_threshold: bool = False
    fixable_by_manual_fusion: bool = False
    score: int = 0
    threshold: int = 0
    shared_bytes: int = 0


@dataclass(frozen=True)
class FusionGroup:
    """A set of operations Inductor fused into one kernel."""

    kernel_name: str
    constituent_buffers: tuple[str, ...] = ()
    constituent_ops: tuple[str, ...] = ()
    shared_bytes: int = 0


@dataclass(frozen=True)
class IntermediateBuffer:
    """Buffer flowing between two kernels — potential fusion target."""

    name: str
    size_bytes: int
    dtype: str
    producer_kernel: str
    consumer_kernels: tuple[str, ...] = ()
    lifetime_span: int = 0
    eliminable_by_fusion: bool = False


@dataclass(frozen=True)
class AutotuneEntry:
    """Inductor autotuning results for a single template operation."""

    operation: str
    num_configs_tried: int = 0
    winner_type: str = ""
    winner_timing_ms: float = 0.0
    runner_up_type: str = ""
    runner_up_timing_ms: float = 0.0
    all_timings: tuple[dict, ...] = ()


@dataclass(frozen=True)
class CodegenStrategy:
    """Codegen-level decisions for a single Inductor kernel."""

    kernel_name: str
    kernel_category: str
    persistent_reduction: bool = False
    cooperative_reduction: bool = False
    no_x_dim: bool = False
    uses_block_pointers: bool = False
    num_reduction_dims: int = 0
    reduction_hint: str = "N/A"
    inductor_block_sizes: dict = field(default_factory=dict)
    inductor_num_warps: int = 4
    inductor_num_stages: int = 1


@dataclass(frozen=True)
class TileSizeAnalysis:
    """Dimensional analysis for tile size grounding."""

    kernel_name: str
    original_shape: tuple[int, ...] = ()
    flattened_numel: int = 0
    contiguous_dim_size: int = 0
    vectorization_max_width: int = 4
    vectorization_aligned: bool = False
    inductor_winner: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPressureInfo:
    """Memory allocation timeline and peak analysis."""

    peak_memory_bytes: int = 0
    peak_kernel: str = ""
    long_lived_buffers: tuple[IntermediateBuffer, ...] = ()


# =========================================================================
# Triton layer
# =========================================================================

@dataclass(frozen=True)
class TritonCompilationInfo:
    """Metadata from Triton's MLIR compilation pipeline."""

    kernel_name: str
    num_warps: int = 4
    num_stages: int = 1
    shared_bytes: int = 0
    n_regs: int = 0
    n_spills: int = 0
    n_max_threads: int = 0
    ttir_text: str = ""
    ttgir_text: str = ""
    compilation_time_us: float = 0.0
    dot_count: int = 0
    load_count: int = 0
    store_count: int = 0
    has_async_copy: bool = False
    has_warp_specialize: bool = False
    mma_type: str = ""


# =========================================================================
# XLA layer (JAX)
# =========================================================================

@dataclass(frozen=True)
class XLAFusionDecision:
    """A single XLA fusion or rejection."""

    step_type: str
    producer_name: str = ""
    consumer_name: str = ""
    reason: str = ""
    us_fused: float = 0.0
    us_unfused: float = 0.0


@dataclass(frozen=True)
class XLAAutotuneResult:
    """XLA autotuning result for a single operation."""

    fusion_name: str
    block_m: int = 0
    block_n: int = 0
    block_k: int = 0
    num_stages: int = 0
    num_warps: int = 0
    run_time_us: float = 0.0


@dataclass
class XLAContext:
    """Compilation context extracted from JAX/XLA."""

    stablehlo_text: str = ""
    optimized_hlo_text: str = ""
    cost_analysis: dict = field(default_factory=dict)
    memory_analysis: dict = field(default_factory=dict)
    fusion_decisions: list[XLAFusionDecision] = field(default_factory=list)
    autotune_results: list[XLAAutotuneResult] = field(default_factory=list)


# =========================================================================
# Top-level CompilerContext
# =========================================================================

@dataclass
class CompilerContext:
    """Complete compiler context extracted from a compilation run.

    All fields are populated during synthesis and are treated as
    read-only thereafter. The pipeline passes this to planner/executor
    via JSON serialization.
    """

    # Metadata
    pytorch_version: str = ""
    extraction_success: bool = False
    extraction_warnings: list[str] = field(default_factory=list)

    # Dynamo layer (pre-decomposition graph)
    dynamo: Optional[DynamoGraphInfo] = None

    # AOTAutograd layer (decomposition paths)
    aot_decomps: list[AOTDecompInfo] = field(default_factory=list)

    # Inductor layer
    ir_nodes: list[IRNodeInfo] = field(default_factory=list)
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    fusion_rejections: list[FusionRejection] = field(default_factory=list)
    intermediate_buffers: list[IntermediateBuffer] = field(default_factory=list)
    autotune_entries: list[AutotuneEntry] = field(default_factory=list)
    codegen_strategies: list[CodegenStrategy] = field(default_factory=list)
    tile_analyses: list[TileSizeAnalysis] = field(default_factory=list)
    memory_pressure: Optional[MemoryPressureInfo] = None

    # Triton layer (per-kernel compilation metadata)
    triton_info: list[TritonCompilationInfo] = field(default_factory=list)

    # XLA layer (JAX — optional)
    xla_context: Optional[XLAContext] = None

    # ---------------------------------------------------------------
    # Query methods
    # ---------------------------------------------------------------

    def get_kernel_ir(self, kernel_name: str) -> list[IRNodeInfo]:
        """Get IR nodes belonging to a specific kernel."""
        group = next(
            (g for g in self.fusion_groups if g.kernel_name == kernel_name),
            None,
        )
        if group is None:
            return []
        return [n for n in self.ir_nodes if n.buffer_name in group.constituent_buffers]

    def get_codegen_strategy(self, kernel_name: str) -> Optional[CodegenStrategy]:
        """Get codegen strategy for a specific kernel."""
        return next(
            (s for s in self.codegen_strategies if s.kernel_name == kernel_name),
            None,
        )

    def get_triton_info(self, kernel_name: str) -> Optional[TritonCompilationInfo]:
        """Get Triton compilation info for a specific kernel."""
        return next(
            (t for t in self.triton_info if t.kernel_name == kernel_name),
            None,
        )

    def get_fusion_opportunities(self) -> list[FusionRejection]:
        """Get near-threshold rejections that could be manually fused."""
        return [
            r for r in self.fusion_rejections
            if r.near_threshold and r.fixable_by_manual_fusion
        ]

    def get_structural_constraints(self, kernel_name: str) -> list[str]:
        """Get things the planner/executor must NOT suggest for this kernel."""
        constraints = []
        strat = self.get_codegen_strategy(kernel_name)
        if strat is None:
            return constraints
        if strat.no_x_dim:
            constraints.append(
                "XBLOCK is structurally fixed at 1 — do NOT suggest increasing XBLOCK"
            )
        if strat.persistent_reduction:
            constraints.append(
                "RBLOCK is a compile-time constant (persistent reduction) — "
                "do NOT add @triton.autotune for RBLOCK"
            )
        if strat.uses_block_pointers:
            constraints.append(
                "Already uses block_ptr/TMA — do NOT suggest converting to block_ptr"
            )
        if strat.cooperative_reduction:
            constraints.append(
                "Already uses cooperative reduction (RSPLIT) — do NOT suggest adding RSPLIT"
            )

        tri = self.get_triton_info(kernel_name)
        if tri:
            if tri.n_regs > 128:
                constraints.append(
                    f"High register pressure ({tri.n_regs} regs/thread) — "
                    "consider reducing registers before adding computation"
                )
            if tri.n_spills > 0:
                constraints.append(
                    f"Register spills detected ({tri.n_spills} spills) — "
                    "reducing register pressure is the top priority"
                )
        return constraints

    def get_autotune_avoid_list(self, kernel_name: str) -> list[dict]:
        """Get configs that Inductor already proved slow (>2x worse than winner)."""
        avoid = []
        for entry in self.autotune_entries:
            if entry.winner_timing_ms <= 0:
                continue
            for timing in entry.all_timings:
                if timing.get("timing_ms", 0) > entry.winner_timing_ms * 2.0:
                    avoid.append(timing.get("config", {}))
        return avoid

    def get_original_op_name(self, kernel_name: str) -> str | None:
        """Map a kernel back to its original high-level op (e.g., layer_norm).

        Uses the decomposition table: if a compound Dynamo op decomposes into
        primitives that match this kernel's IR node ops, returns the compound op.
        """
        if self.dynamo is None or not self.dynamo.original_ops:
            return None

        ir_nodes = self.get_kernel_ir(kernel_name)
        if not ir_nodes:
            strat = self.get_codegen_strategy(kernel_name)
            if strat and "_fused_" in kernel_name:
                fused_part = kernel_name.split("_fused_", 1)[-1]
                for op in self.dynamo.original_ops:
                    if op.is_compound and op.op_name.lower() in fused_part:
                        return op.op_name
            return None

        kernel_op_names = {n.node_type.lower() for n in ir_nodes if n.node_type}

        for decomp in self.aot_decomps:
            decomp_ops = {d.lower() for d in decomp.decomposed_into}
            if decomp_ops and decomp_ops.issubset(kernel_op_names):
                return decomp.original_op

        for op in self.dynamo.original_ops:
            if op.is_compound and op.op_name.lower() in kernel_name.lower():
                return op.op_name

        return None

    # ---------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON for subprocess communication and checkpointing."""
        import dataclasses

        def _serialize(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(dataclasses.asdict(self), default=_serialize, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> CompilerContext:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        dynamo = None
        if data.get("dynamo"):
            d = data["dynamo"]
            dynamo = DynamoGraphInfo(
                original_ops=tuple(DynamoOpInfo(**op) for op in d.get("original_ops", [])),
                graph_breaks=tuple(GraphBreakInfo(**gb) for gb in d.get("graph_breaks", [])),
                num_subgraphs=d.get("num_subgraphs", 1),
                unsupported_ops=tuple(d.get("unsupported_ops", [])),
            )

        aot_decomps = [AOTDecompInfo(**ad) for ad in data.get("aot_decomps", [])]

        ir_nodes = []
        for n in data.get("ir_nodes", []):
            red = None
            if n.get("reduction"):
                red = ReductionInfo(**n["reduction"])
            n_copy = {k: v for k, v in n.items() if k != "reduction"}
            ir_nodes.append(IRNodeInfo(reduction=red, **n_copy))

        fusion_groups = [FusionGroup(**fg) for fg in data.get("fusion_groups", [])]
        fusion_rejections = [FusionRejection(**fr) for fr in data.get("fusion_rejections", [])]
        intermediate_buffers = [IntermediateBuffer(**ib) for ib in data.get("intermediate_buffers", [])]
        autotune_entries = [AutotuneEntry(**ae) for ae in data.get("autotune_entries", [])]
        codegen_strategies = [CodegenStrategy(**cs) for cs in data.get("codegen_strategies", [])]
        tile_analyses = [TileSizeAnalysis(**ta) for ta in data.get("tile_analyses", [])]
        triton_info = [TritonCompilationInfo(**ti) for ti in data.get("triton_info", [])]

        mp = None
        if data.get("memory_pressure"):
            mp_data = data["memory_pressure"]
            mp = MemoryPressureInfo(
                peak_memory_bytes=mp_data.get("peak_memory_bytes", 0),
                peak_kernel=mp_data.get("peak_kernel", ""),
                long_lived_buffers=tuple(
                    IntermediateBuffer(**lb) for lb in mp_data.get("long_lived_buffers", [])
                ),
            )

        xla = None
        if data.get("xla_context"):
            xd = data["xla_context"]
            xla = XLAContext(
                stablehlo_text=xd.get("stablehlo_text", ""),
                optimized_hlo_text=xd.get("optimized_hlo_text", ""),
                cost_analysis=xd.get("cost_analysis", {}),
                memory_analysis=xd.get("memory_analysis", {}),
                fusion_decisions=[XLAFusionDecision(**fd) for fd in xd.get("fusion_decisions", [])],
                autotune_results=[XLAAutotuneResult(**ar) for ar in xd.get("autotune_results", [])],
            )

        return cls(
            pytorch_version=data.get("pytorch_version", ""),
            extraction_success=data.get("extraction_success", False),
            extraction_warnings=data.get("extraction_warnings", []),
            dynamo=dynamo,
            aot_decomps=aot_decomps,
            ir_nodes=ir_nodes,
            fusion_groups=fusion_groups,
            fusion_rejections=fusion_rejections,
            intermediate_buffers=intermediate_buffers,
            autotune_entries=autotune_entries,
            codegen_strategies=codegen_strategies,
            tile_analyses=tile_analyses,
            memory_pressure=mp,
            triton_info=triton_info,
            xla_context=xla,
        )
