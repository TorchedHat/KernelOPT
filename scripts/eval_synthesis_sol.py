#!/usr/bin/env python3
"""Evaluate Feature 4 synthesis on SOL-ExecBench problems.

Converts SOL-ExecBench JSON specs into .py files, then runs synthesis.

Usage:
    python eval_synthesis_sol.py --split train --gpu 2
    python eval_synthesis_sol.py --split L1 --gpu 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOL_ROOT = Path(__file__).parent / "tests_suite" / "benchmarks" / "sol_execbench"
KERNELOPT_PKG = Path(__file__).parent / "kernelopt"
RESULTS_DIR = Path(__file__).parent / "eval_synthesis_results"
TEMP_DIR = Path("/tmp/sol_eval_py")

sys.path.insert(0, str(KERNELOPT_PKG))


_DTYPE_MAP = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "int8": "torch.int8",
    "uint8": "torch.uint8",
    "bool": "torch.bool",
    "fp8_e4m3": "torch.float8_e4m3fn",
    "fp8_e5m2": "torch.float8_e5m2",
}


def sol_json_to_py(json_path: str) -> str:
    """Convert a SOL-ExecBench JSON spec to a standalone .py file."""
    data = json.load(open(json_path))
    reference = data.get("reference", "")
    if not reference:
        raise ValueError(f"No reference code in {json_path}")

    inputs_spec = json.loads(data.get("inputs", "{}"))
    axes_spec = json.loads(data.get("axes", "{}"))
    workloads = json.loads(data.get("workloads", "[]"))

    axis_values = {}
    for name, spec in axes_spec.items():
        if isinstance(spec, dict) and spec.get("type") == "const":
            axis_values[name] = spec["value"]

    if workloads:
        wl = workloads[0] if isinstance(workloads, list) else workloads
        wl_axes = wl.get("axes", {}) if isinstance(wl, dict) else {}
        for name, val in wl_axes.items():
            if name not in axis_values:
                axis_values[name] = val

    for name in axes_spec:
        if name not in axis_values:
            axis_values[name] = 64

    get_inputs_lines = ["def get_inputs():"]
    input_names = []

    for inp_name, inp_spec in inputs_spec.items():
        if not isinstance(inp_spec, dict):
            continue

        shape = inp_spec.get("shape")
        dtype = inp_spec.get("dtype", "float32")
        torch_dtype = _DTYPE_MAP.get(dtype, "torch.float32")

        if shape is None:
            if "float" in dtype or dtype in ("bfloat16",):
                get_inputs_lines.append(f"    {inp_name} = torch.tensor(0.1, dtype={torch_dtype}, device='cuda')")
            elif "int" in dtype:
                get_inputs_lines.append(f"    {inp_name} = torch.tensor(1, dtype={torch_dtype}, device='cuda')")
            elif dtype == "bool":
                get_inputs_lines.append(f"    {inp_name} = torch.tensor(True, device='cuda')")
            else:
                get_inputs_lines.append(f"    {inp_name} = torch.tensor(0.1, dtype={torch_dtype}, device='cuda')")
            input_names.append(inp_name)
            continue

        if not isinstance(shape, list):
            get_inputs_lines.append(f"    {inp_name} = torch.tensor(0.1, dtype={torch_dtype}, device='cuda')")
            input_names.append(inp_name)
            continue

        shape_parts = []
        for dim in shape:
            if isinstance(dim, int):
                shape_parts.append(str(dim))
            elif isinstance(dim, str) and dim in axis_values:
                shape_parts.append(str(axis_values[dim]))
            elif isinstance(dim, str):
                shape_parts.append("64")
            else:
                shape_parts.append("64")
        shape_str = ", ".join(shape_parts)

        if dtype == "bool":
            get_inputs_lines.append(
                f"    {inp_name} = torch.randint(0, 2, ({shape_str},), device='cuda').bool()"
            )
        elif "int" in dtype or dtype in ("uint8",):
            get_inputs_lines.append(
                f"    {inp_name} = torch.randint(0, 10, ({shape_str},), dtype={torch_dtype}, device='cuda')"
            )
        elif "fp8" in dtype:
            get_inputs_lines.append(
                f"    {inp_name} = torch.randn({shape_str}, device='cuda').to({torch_dtype})"
            )
        else:
            get_inputs_lines.append(
                f"    {inp_name} = torch.randn({shape_str}, dtype={torch_dtype}, device='cuda')"
            )
        input_names.append(inp_name)

    ret = ", ".join(input_names)
    get_inputs_lines.append(f"    return [{ret}]")

    py_code = reference.strip() + "\n\n" + "\n".join(get_inputs_lines) + "\n"

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(json_path).stem
    out_path = TEMP_DIR / f"{stem}.py"
    out_path.write_text(py_code)
    return str(out_path)


def run_one(problem_json: str, gpu_id: int):
    """Run synthesis on a single SOL-ExecBench problem."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch, SynthesisError
    from kernelopt.synthesis.wrap_kernel import wrap_inductor_output

    data = json.load(open(problem_json))
    result = {
        "problem": Path(problem_json).stem,
        "problem_path": str(problem_json),
        "name": data.get("name", ""),
        "gpu": gpu_id,
        "status": "unknown",
        "n_triton_kernels": 0,
        "has_extern_kernels": False,
        "error": "",
        "compile_time_s": 0.0,
        "wrap_time_s": 0.0,
        "exec_time_s": 0.0,
        "total_time_s": 0.0,
        "exec_ok": False,
    }

    t_total_start = time.time()

    try:
        py_path = sol_json_to_py(problem_json)
    except Exception as e:
        result["status"] = "convert_error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["total_time_s"] = round(time.time() - t_total_start, 2)
        return result

    src = Path(py_path).read_text()

    t0 = time.time()
    try:
        extraction_results = extract_triton_from_pytorch(py_path)
    except SynthesisError as e:
        result["compile_time_s"] = round(time.time() - t0, 2)
        result["total_time_s"] = round(time.time() - t_total_start, 2)
        result["status"] = "synthesis_error"
        result["error"] = str(e)[:500]
        return result
    except Exception as e:
        result["compile_time_s"] = round(time.time() - t0, 2)
        result["total_time_s"] = round(time.time() - t_total_start, 2)
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result
    result["compile_time_s"] = round(time.time() - t0, 2)

    if not extraction_results:
        result["total_time_s"] = round(time.time() - t_total_start, 2)
        result["status"] = "synthesis_error"
        result["error"] = "torch.compile returned empty source list"
        return result

    primary = extraction_results[0]
    n_kernels = len(primary["triton_kernels"])
    has_extern = "extern_kernels" in primary["inductor_source"]
    result["n_triton_kernels"] = n_kernels
    result["has_extern_kernels"] = has_extern

    t1 = time.time()
    try:
        wrapped = wrap_inductor_output(
            inductor_source=primary["inductor_source"],
            original_pytorch_source=src,
            triton_kernel_sources=primary["triton_kernels"],
        )
    except Exception as e:
        result["wrap_time_s"] = round(time.time() - t1, 2)
        result["total_time_s"] = round(time.time() - t_total_start, 2)
        result["status"] = "wrap_error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result
    result["wrap_time_s"] = round(time.time() - t1, 2)

    outpath = f"/tmp/eval_sol_{Path(problem_json).stem}.py"
    Path(outpath).write_text(wrapped)

    t2 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, outpath, "--profile-only"],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        result["exec_time_s"] = round(time.time() - t2, 2)
        if r.returncode == 0:
            result["status"] = "pass"
            result["exec_ok"] = True
        else:
            err_lines = (r.stderr or r.stdout).strip().split("\n")
            last_err = err_lines[-1][:300] if err_lines else "?"
            result["status"] = "exec_fail"
            result["error"] = last_err
    except Exception as e:
        result["exec_time_s"] = round(time.time() - t2, 2)
        result["status"] = "exec_error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    result["total_time_s"] = round(time.time() - t_total_start, 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="train, L1, L2, Quant, FlashInfer-Bench")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999)
    args = parser.parse_args()

    split_dir = SOL_ROOT / args.split / "train" if args.split != "train" else SOL_ROOT / "train"
    if not split_dir.exists():
        print(f"ERROR: {split_dir} does not exist")
        sys.exit(1)

    problems = sorted(split_dir.glob("kernel_*.json"))
    problems = [p for p in problems if args.start <= int(p.stem.split("_")[1]) <= args.end]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"sol_{args.split}_gpu{args.gpu}.json"

    print(f"{'='*70}")
    print(f"  SOL-ExecBench [{args.split}] Synthesis Evaluation")
    print(f"  Problems: {len(problems)} | GPU: {args.gpu} | No timeouts")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")
    print()

    all_results = []
    counts = {}
    total = len(problems)
    wall_start = time.time()

    for i, problem in enumerate(problems, 1):
        kid = problem.stem
        data = json.load(open(problem))
        name = data.get("name", "")[:40]
        print(f"[{i:3d}/{total}] {kid} ({name}) ... ", end="", flush=True)

        result = run_one(str(problem), args.gpu)
        all_results.append(result)

        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        ct = result["compile_time_s"]
        et = result["exec_time_s"]
        tt = result["total_time_s"]

        if status == "pass":
            print(f"PASS  compile={ct:.1f}s exec={et:.1f}s total={tt:.1f}s  [{result['n_triton_kernels']}k]")
        else:
            print(f"{status.upper()}  {tt:.1f}s  {result['error'][:60]}")

        with open(out_file, "w") as f:
            json.dump({
                "split": args.split,
                "gpu": args.gpu,
                "total": total,
                "completed": i,
                "wall_time_s": round(time.time() - wall_start, 1),
                "counts": counts,
                "results": all_results,
            }, f, indent=2)

    wall_total = time.time() - wall_start
    print()
    print(f"{'='*70}")
    print(f"  SUMMARY: SOL-ExecBench [{args.split}] ({total} problems, {wall_total:.0f}s)")
    print(f"{'='*70}")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * c / total if total > 0 else 0
        print(f"  {s:20s}: {c:3d}/{total} ({pct:5.1f}%)")

    ct_list = [r["compile_time_s"] for r in all_results if r["compile_time_s"] > 0]
    et_list = [r["exec_time_s"] for r in all_results if r["exec_time_s"] > 0]
    if ct_list:
        print(f"  Compile: avg={sum(ct_list)/len(ct_list):.1f}s max={max(ct_list):.1f}s")
    if et_list:
        print(f"  Exec:    avg={sum(et_list)/len(et_list):.1f}s max={max(et_list):.1f}s")
    print(f"  Wall:    {wall_total:.0f}s ({wall_total/60:.1f}min)")
    print(f"\n  Results: {out_file}")


if __name__ == "__main__":
    main()
