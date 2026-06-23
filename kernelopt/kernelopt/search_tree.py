"""Search tree over kernel optimization candidates.

Instead of greedy hill-climbing (always optimize from the single best),
maintain a tree of candidates where any leaf can be expanded. This
enables backtracking and exploring alternative optimization paths.

Each node stores a kernel variant with its profiled latency. The
selection policy decides which nodes to expand in the next iteration.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SearchNode:
    id: str
    parent_id: str | None
    kernel_source: str
    latency_ns: float
    speedup: float  # vs baseline (root)
    direction: str  # optimization applied to get here
    iteration: int
    correctness: bool
    plan_letter: str = ""
    children: list[str] = field(default_factory=list)
    n_expansions: int = 0  # how many times this node was selected for expansion


class SearchTree:
    """Tree of kernel optimization candidates with selection policies."""

    def __init__(self, path: str | Path | None = None):
        self.nodes: dict[str, SearchNode] = {}
        self.root_id: str | None = None
        self._baseline_ns: float = float("inf")
        self._path = Path(path) if path else None

        if self._path and self._path.exists():
            self._load()

    def set_root(self, kernel_source: str, latency_ns: float) -> str:
        """Create the root node (baseline kernel)."""
        root_id = "root"
        self.root_id = root_id
        self._baseline_ns = latency_ns
        self.nodes[root_id] = SearchNode(
            id=root_id,
            parent_id=None,
            kernel_source=kernel_source,
            latency_ns=latency_ns,
            speedup=1.0,
            direction="baseline",
            iteration=0,
            correctness=True,
        )
        self._save()
        return root_id

    def add_candidate(
        self,
        parent_id: str,
        kernel_source: str,
        latency_ns: float,
        direction: str,
        iteration: int,
        correctness: bool,
        plan_letter: str = "",
    ) -> str:
        """Add a profiled candidate as a child of parent_id."""
        node_id = uuid.uuid4().hex[:8]
        speedup = round(self._baseline_ns / latency_ns, 4) if latency_ns > 0 else 0.0

        node = SearchNode(
            id=node_id,
            parent_id=parent_id,
            kernel_source=kernel_source,
            latency_ns=latency_ns,
            speedup=speedup,
            direction=direction,
            iteration=iteration,
            correctness=correctness,
            plan_letter=plan_letter,
        )
        self.nodes[node_id] = node

        if parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)

        self._save()
        return node_id

    def select_nodes(self, policy: str = "greedy", n: int = 1) -> list[SearchNode]:
        """Select nodes to expand in the next iteration.

        Policies:
          greedy — the single best node (lowest latency)
          top_k  — the top-N nodes by latency
          ucb    — Upper Confidence Bound (explore vs exploit)
        """
        leaves = [
            node for node in self.nodes.values()
            if node.correctness and node.latency_ns < float("inf")
            and not node.children  # only unexpanded nodes
        ]
        if not leaves:
            return []

        if policy == "greedy":
            best = min(leaves, key=lambda n: n.latency_ns)
            best.n_expansions += 1
            selected = [best]
        elif policy == "top_k":
            ranked = sorted(leaves, key=lambda n: n.latency_ns)
            selected = ranked[:n]
            for node in selected:
                node.n_expansions += 1
        elif policy == "ucb":
            total_expansions = sum(n.n_expansions for n in leaves) + 1
            c = 1.4  # exploration constant

            def ucb_score(node: SearchNode) -> float:
                exploit = node.speedup
                explore = c * math.sqrt(
                    math.log(total_expansions) / (node.n_expansions + 1)
                )
                return exploit + explore

            ranked = sorted(leaves, key=ucb_score, reverse=True)
            selected = ranked[:n]
            for node in selected:
                node.n_expansions += 1
        else:
            # Unknown policy — fall back to greedy (recurse, which saves)
            return self.select_nodes("greedy", n)

        # Persist updated n_expansions so UCB state survives across iterations
        self._save()
        return selected

    def get_best(self) -> SearchNode | None:
        """Return the node with lowest latency (global best)."""
        valid = [
            n for n in self.nodes.values()
            if n.correctness and n.latency_ns < float("inf")
        ]
        return min(valid, key=lambda n: n.latency_ns) if valid else None

    def get_node(self, node_id: str) -> SearchNode | None:
        return self.nodes.get(node_id)

    def stats(self) -> dict:
        """Summary statistics for logging."""
        total = len(self.nodes)
        correct = sum(1 for n in self.nodes.values() if n.correctness)
        best = self.get_best()
        depths = {}
        for node in self.nodes.values():
            d = self._depth(node.id)
            depths[d] = depths.get(d, 0) + 1

        return {
            "total_nodes": total,
            "correct_nodes": correct,
            "best_speedup": best.speedup if best else 0,
            "best_latency_ns": best.latency_ns if best else float("inf"),
            "depth_distribution": depths,
        }

    def format_tree(self, max_depth: int = 4) -> str:
        """Format the tree as a human-readable string for logging."""
        if not self.root_id:
            return "(empty tree)"

        lines = []
        self._format_node(self.root_id, 0, max_depth, lines)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _depth(self, node_id: str) -> int:
        depth = 0
        current = node_id
        while current and current in self.nodes:
            parent = self.nodes[current].parent_id
            if parent is None:
                break
            current = parent
            depth += 1
        return depth

    def _format_node(
        self, node_id: str, depth: int, max_depth: int, lines: list,
    ) -> None:
        if depth > max_depth:
            return
        node = self.nodes.get(node_id)
        if not node:
            return

        indent = "  " * depth
        marker = "+" if node.correctness else "x"
        lat = f"{node.latency_ns/1000:.1f}µs" if node.latency_ns < float("inf") else "fail"
        lines.append(
            f"{indent}[{marker}] {node.direction[:40]} — {lat} ({node.speedup:.2f}x)"
        )
        for child_id in node.children:
            self._format_node(child_id, depth + 1, max_depth, lines)

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "root_id": self.root_id,
            "baseline_ns": self._baseline_ns,
            "nodes": {
                nid: {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "latency_ns": n.latency_ns,
                    "speedup": n.speedup,
                    "direction": n.direction,
                    "iteration": n.iteration,
                    "correctness": n.correctness,
                    "plan_letter": n.plan_letter,
                    "children": n.children,
                    "n_expansions": n.n_expansions,
                    # kernel_source stored separately to keep JSON small
                }
                for nid, n in self.nodes.items()
            },
        }
        self._path.write_text(json.dumps(data, indent=2))

        # Save kernel sources in a sibling directory
        kernel_dir = self._path.parent / "tree_kernels"
        kernel_dir.mkdir(exist_ok=True)
        for nid, node in self.nodes.items():
            kpath = kernel_dir / f"{nid}.py"
            if not kpath.exists() and node.kernel_source:
                kpath.write_text(node.kernel_source)

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self.root_id = data.get("root_id")
        self._baseline_ns = data.get("baseline_ns", float("inf"))

        kernel_dir = self._path.parent / "tree_kernels"
        for nid, nd in data.get("nodes", {}).items():
            kernel_source = ""
            kpath = kernel_dir / f"{nid}.py"
            if kpath.exists():
                kernel_source = kpath.read_text()

            self.nodes[nid] = SearchNode(
                id=nd["id"],
                parent_id=nd.get("parent_id"),
                kernel_source=kernel_source,
                latency_ns=nd.get("latency_ns", float("inf")),
                speedup=nd.get("speedup", 0),
                direction=nd.get("direction", ""),
                iteration=nd.get("iteration", 0),
                correctness=nd.get("correctness", False),
                plan_letter=nd.get("plan_letter", ""),
                children=nd.get("children", []),
                n_expansions=nd.get("n_expansions", 0),
            )
