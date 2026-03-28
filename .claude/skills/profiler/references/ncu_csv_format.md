# NCU Sections-Based CSV Format Reference

This file documents the structure of the Nsight Compute section-based CSV export.
It covers parsing mechanics only — no optimization rules or verdict logic.

---

## CSV columns

```
ID                   — integer, groups all rows for one kernel launch (0-indexed)
Process ID           — OS process ID
Process Name         — e.g. "python3.10"
Host Name            — machine hostname
Kernel Name          — full mangled C++ symbol name
Context              — CUDA context ID
Stream               — CUDA stream ID
Block Size           — e.g. "(256, 1, 1)"
Grid Size            — e.g. "(1056, 1, 1)"
Device               — device index
CC                   — compute capability, e.g. "9.0"
Section Name         — for metric rows: section group name; for rule rows: variant name
Metric Name          — metric label (empty for rule rows)
Metric Unit          — unit string, e.g. "%", "ns", "cycle", "byte/s" (empty for rule rows)
Metric Value         — numeric or string value (empty for rule rows)
Rule Name            — rule identifier (empty for metric rows)
Rule Type            — "OPT", "INF", or "WRN" (empty for metric rows)
Rule Description     — full natural language description from NCU (empty for metric rows)
Estimated Speedup Type — "global", "local", or empty
Estimated Speedup    — float percentage or empty
```

---

## Row types

### Metric rows
`Metric Name` is populated; `Rule Name` is empty.
Extract: Section Name + Metric Name + Metric Unit + Metric Value.

### Rule rows
`Rule Name` is populated; `Metric Name` / `Metric Value` are empty.
The `Section Name` for rule rows is a variant of the section (e.g. `SpeedOfLight` for
SOL rules, `WarpStateStats` for warp stall rules) — different from metric row section names.

---

## Sections present in a full NCU export

These are the `Section Name` values for metric rows:

| Section Name | What it covers |
|---|---|
| GPU Speed Of Light Throughput | Top-level throughput %, duration, frequencies |
| Compute Workload Analysis | IPC, SM busy, issue slot utilization |
| Memory Workload Analysis | L1/L2 hit rates, bandwidth, spilling |
| Occupancy | Theoretical and achieved occupancy, block limits |
| Scheduler Statistics | Eligible/active warps per scheduler |
| Warp State Statistics | CPI, active threads per warp |
| Launch Statistics | Block/grid size, registers, shared memory, SM count |
| Instruction Statistics | Executed instructions, spilling |
| Source Counters | Branch efficiency, divergent branches |
| GPU and Memory Workload Distribution | Average/total cycles across L1, L2, DRAM, SM |
| PM Sampling | Sampling configuration metadata |

Rule row `Section Name` variants include: `SpeedOfLight`, `SpeedOfLight_RooflineChart`,
`ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis_Chart`, `MemoryWorkloadAnalysis_Tables`,
`WarpStateStats`, `InstructionStats`, `LaunchStats`, `SchedulerStats`, `Occupancy`,
`WorkloadDistribution`, `SourceCounters`, `PmSampling`.

---

## Parsing code (Python)

```python
import csv
from collections import defaultdict

def parse_ncu_csv(filepath, kernel_id=0):
    """
    Returns:
        sections: dict of {section_name: {metric_name: {"value": v, "unit": u}}}
        rules:    list of dicts with keys: name, type, description, speedup_type, speedup_pct
        launch:   dict of launch configuration values
    """
    sections = defaultdict(dict)
    rules = []

    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_id = int(row.get('ID', 0))
            except ValueError:
                continue
            if row_id != kernel_id:
                continue

            section     = row['Section Name'].strip()
            metric_name = row['Metric Name'].strip()
            rule_name   = row['Rule Name'].strip()

            if metric_name:
                raw = row['Metric Value'].strip().replace(',', '')
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
                sections[section][metric_name] = {
                    "value": val,
                    "unit": row['Metric Unit'].strip()
                }

            elif rule_name:
                sp_raw = row['Estimated Speedup'].strip()
                rules.append({
                    "name":         rule_name,
                    "type":         row['Rule Type'].strip(),
                    "description":  row['Rule Description'].strip(),
                    "speedup_type": row['Estimated Speedup Type'].strip(),
                    "speedup_pct":  float(sp_raw) if sp_raw else None,
                })

    # Extract launch config convenience dict
    launch_sec = sections.get('Launch Statistics', {})
    launch = {k: v['value'] for k, v in launch_sec.items()}

    return dict(sections), rules, launch


def get_kernel_ids(filepath):
    """List all (ID, Kernel Name) pairs in a multi-kernel CSV."""
    seen = {}
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kid = row.get('ID', '0').strip()
            kname = row.get('Kernel Name', '').strip()
            if kid not in seen and kname:
                seen[kid] = kname
    return [(int(k), v) for k, v in sorted(seen.items())]
```

---

## Framework detection patterns

Inspect the full `Kernel Name` string:

| Pattern in kernel name | Framework |
|---|---|
| `triton_` prefix, `__triton`, `triton::` | Triton |
| `helion_` prefix, `helion::` | Helion |
| `at::` namespace (PyTorch ATen) | CUDA (PyTorch compiled) |
| `__global__` in name, CUDA mangling (`void kernel_name<`) | CUDA |
| `pycuda_` prefix | CUDA (PyCUDA) |
| `jit_` prefix with numba patterns | Numba CUDA |
| Anything else | unknown |

This is informational only — pass it as `framework_hint` to the planner.

---

## Notes on parsing edge cases

- Values may use comma thousands separators (e.g. `"1,234,567"`) — strip commas before parsing
- The file may have a UTF-8 BOM — use `encoding='utf-8-sig'`
- `Metric Value` is sometimes a non-numeric string (e.g. `"PolicySpread"`, `"all"`) — keep as string
- `Block Size` and `Grid Size` are strings like `"(256, 1, 1)"` — parse if needed or keep as string
- In multi-kernel files, the same `(Section Name, Metric Name)` pair may appear once per kernel ID
- The `L2 Hit Rate` can exceed 100% due to prefetch counting artifacts — report as-is
