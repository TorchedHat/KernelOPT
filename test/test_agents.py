"""
Test each agent independently using direct transformers inference (no vLLM, no TorchServe).

Usage:
    python test/test_agents.py --agent profiler
    python test/test_agents.py --agent planner --gpu 2
    python test/test_agents.py --agent executor --gpu 2
    python test/test_agents.py --agent summarizer --gpu 2
    python test/test_agents.py --agent all --gpu 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))

# ---------------------------------------------------------------------------
# Direct LLM client (no server, loads model in-process)
# ---------------------------------------------------------------------------

_LOADED_MODELS: dict[str, tuple] = {}
_MODEL_LOCK = threading.Lock()


def load_model(model_name: str, gpu_id: int = 2):
    """Load a model+tokenizer once, cache for reuse. Thread-safe."""
    with _MODEL_LOCK:
        if model_name in _LOADED_MODELS:
            return _LOADED_MODELS[model_name]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n  Loading {model_name} on GPU {gpu_id}...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if "FP8" not in model_name else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map={"": f"cuda:{gpu_id}"},
            trust_remote_code=True,
        )
        model.eval()

        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s")

        _LOADED_MODELS[model_name] = (model, tokenizer)
        return model, tokenizer


_INFERENCE_LOCK = threading.Lock()


def direct_inference(
    model_name: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.2,
    gpu_id: int = 2,
) -> str:
    """Run inference directly with transformers (no server). Thread-safe."""
    import torch

    model, tokenizer = load_model(model_name, gpu_id)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    with _INFERENCE_LOCK:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return generated


# ---------------------------------------------------------------------------
# Monkey-patch build_llm to use direct inference
# ---------------------------------------------------------------------------

def patch_build_llm(gpu_id: int):
    """Replace build_llm so agents use direct transformers inference."""
    from unittest.mock import MagicMock

    import kernelopt.llm as llm_module

    original_build_llm = llm_module.build_llm

    def patched_build_llm(model=None, api_key=None, max_tokens=4096,
                          temperature=0.2, base_url=None):
        model = model or "Qwen/Qwen3-30B-A3B"

        mock_llm = MagicMock()

        def invoke(messages):
            msg_dicts = []
            for m in messages:
                if hasattr(m, "type"):
                    role = {"system": "system", "human": "user", "ai": "assistant"}.get(m.type, "user")
                else:
                    role = "user"
                msg_dicts.append({"role": role, "content": m.content})

            result = direct_inference(
                model, msg_dicts, max_tokens=max_tokens,
                temperature=temperature, gpu_id=gpu_id,
            )

            response = MagicMock()
            response.content = result
            return response

        mock_llm.invoke = invoke
        return mock_llm

    llm_module.build_llm = patched_build_llm
    return original_build_llm


# ---------------------------------------------------------------------------
# Test: Profiler
# ---------------------------------------------------------------------------

def test_profiler():
    """Test profiler agent: run NCU on softmax kernel, build profiling context."""
    print("\n" + "=" * 60)
    print("  TEST: Profiler Agent")
    print("=" * 60)

    from kernelopt.agents.profiler import build_profiling_context, get_latency_ns, run_ncu
    from kernelopt.functions.csv_parser import parse_ncu_csv

    kernel_path = str(Path(__file__).parent / "softmaxtri.py")
    output_base = "/tmp/kernelopt_test_profiler"

    print(f"\n  Profiling {kernel_path}...")
    t0 = time.time()
    csv_path, ncu_rep = run_ncu(
        kernel_path, output_base,
        kernel_args=["--profile-only"],
    )
    elapsed = time.time() - t0
    print(f"  NCU completed in {elapsed:.1f}s")
    print(f"  CSV: {csv_path}")
    print(f"  ncu-rep: {ncu_rep}")

    sections, rules, launch = parse_ncu_csv(csv_path)
    ctx = build_profiling_context(sections, rules, launch, kernel_path)
    latency = get_latency_ns(ctx)

    print(f"\n  Kernel: {ctx['kernel_name']}")
    print(f"  Framework: {ctx['framework_hint']}")
    print(f"  Latency: {latency:.0f} ns")
    print(f"  Rules: {len(ctx['rules'])}")

    for r in sorted(ctx["rules"], key=lambda x: x.get("speedup_pct") or 0, reverse=True)[:3]:
        print(f"    - {r['name']}: {r.get('speedup_pct', '?')}% ({r['type']})")

    assert ctx["kernel_name"], "kernel_name should not be empty"
    assert latency > 0 and latency < float("inf"), f"Invalid latency: {latency}"
    assert len(ctx["rules"]) > 0, "Should have at least one rule"

    print("\n  PASSED: Profiler agent works correctly")
    return ctx


# ---------------------------------------------------------------------------
# Test: Planner
# ---------------------------------------------------------------------------

def test_planner(profiling_ctx: dict, gpu_id: int):
    """Test planner agent: generate an optimization plan from profiling data."""
    print("\n" + "=" * 60)
    print("  TEST: Planner Agent")
    print("=" * 60)

    from kernelopt.agents.planner import run_planner

    kernel_source = Path(Path(__file__).parent / "softmaxtri.py").read_text()

    print("\n  Generating optimization plan...")
    t0 = time.time()
    plan = run_planner(
        profiling_context=profiling_ctx,
        kernel_source=kernel_source,
        optimization_memory=[],
        model="Qwen/Qwen3-30B-A3B",
        plan_index=0,
        total_plans=1,
    )
    elapsed = time.time() - t0
    print(f"  Plan generated in {elapsed:.1f}s")

    print(f"\n  Plan ID: {plan.get('plan_id', '?')}")
    print(f"  Direction: {plan.get('direction', '?')}")
    print(f"  Framework: {plan.get('framework', '?')}")
    print(f"  Confidence: {plan.get('confidence', '?')}")
    print(f"  Action: {plan.get('step', {}).get('action', '?')}")
    print(f"  Target: {plan.get('step', {}).get('target', '?')}")
    print(f"  Change: {plan.get('step', {}).get('change', '?')[:100]}")

    assert plan.get("direction"), "Plan should have a direction"
    assert plan.get("step"), "Plan should have a step"
    assert plan.get("confidence") in ("high", "medium", "low"), f"Bad confidence: {plan.get('confidence')}"

    print("\n  PASSED: Planner agent works correctly")
    return plan


# ---------------------------------------------------------------------------
# Test: Executor
# ---------------------------------------------------------------------------

def test_executor(plan: dict, gpu_id: int):
    """Test executor agent: apply an optimization plan to produce optimized code."""
    print("\n" + "=" * 60)
    print("  TEST: Executor Agent")
    print("=" * 60)

    from kernelopt.agents.executor import run_executor

    kernel_source = Path(Path(__file__).parent / "softmaxtri.py").read_text()

    print("\n  Applying optimization plan...")
    t0 = time.time()
    optimized_source, execution_result = run_executor(
        kernel_source=kernel_source,
        optimization_plan=plan,
        framework=plan.get("framework", "triton"),
        model="Qwen/Qwen3-30B-A3B",
    )
    elapsed = time.time() - t0
    print(f"  Executor completed in {elapsed:.1f}s")

    print(f"\n  Status: {execution_result.get('status', '?')}")
    print(f"  Lines changed: {execution_result.get('lines_changed', '?')}")
    print(f"  Ready for profiling: {execution_result.get('ready_for_profiling', '?')}")
    print(f"  Output length: {len(optimized_source)} chars")

    # Show first diff
    orig_lines = set(kernel_source.splitlines())
    new_lines = set(optimized_source.splitlines())
    added = new_lines - orig_lines
    if added:
        print(f"\n  Sample new lines:")
        for line in list(added)[:5]:
            print(f"    + {line[:100]}")

    assert len(optimized_source) > 100, "Optimized source too short"
    assert execution_result.get("status") == "success", f"Status: {execution_result.get('status')}"

    print("\n  PASSED: Executor agent works correctly")
    return optimized_source, execution_result


# ---------------------------------------------------------------------------
# Test: Summarizer
# ---------------------------------------------------------------------------

def test_summarizer(profiling_ctx: dict, plan: dict, optimized_source: str, gpu_id: int):
    """Test summarizer agent: produce experience item from before/after data."""
    print("\n" + "=" * 60)
    print("  TEST: Summarizer Agent")
    print("=" * 60)

    from kernelopt.agents.summarizer import run_summarizer

    kernel_source = Path(Path(__file__).parent / "softmaxtri.py").read_text()

    # Use same profiling_ctx for before/after (just testing the agent interface)
    print("\n  Summarizing optimization result...")
    t0 = time.time()
    experience_item, memory_update = run_summarizer(
        profiling_before=profiling_ctx,
        profiling_after=profiling_ctx,  # same for test purposes
        optimization_plan=plan,
        kernel_before=kernel_source,
        kernel_after=optimized_source,
        iteration=1,
        framework=plan.get("framework", "triton"),
        speedup=1.05,
        existing_memory=[],
        model="Qwen/Qwen3-30B-A3B",
    )
    elapsed = time.time() - t0
    print(f"  Summarizer completed in {elapsed:.1f}s")

    print(f"\n  Item ID: {experience_item.get('item_id', '?')}")
    print(f"  Direction: {experience_item.get('direction', '?')}")
    print(f"  Rewrite type: {experience_item.get('rewrite_type', '?')}")
    print(f"  Strategy: {experience_item.get('strategy_title', '?')}")
    print(f"  Memory action: {memory_update.get('action', '?')}")
    print(f"  Memory reason: {memory_update.get('reason', '?')[:100]}")

    assert experience_item.get("item_id"), "Should have item_id"
    assert experience_item.get("rewrite_type") in ("positive", "negative"), \
        f"Bad rewrite_type: {experience_item.get('rewrite_type')}"
    assert memory_update.get("action") in ("append", "replace", "skip"), \
        f"Bad action: {memory_update.get('action')}"

    print("\n  PASSED: Summarizer agent works correctly")
    return experience_item


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test kernelopt agents individually")
    parser.add_argument("--agent", choices=["profiler", "planner", "executor", "summarizer", "all"],
                        default="all", help="Which agent to test")
    parser.add_argument("--gpu", type=int, default=2, help="GPU device ID")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    # Patch LLM for agents that need it
    if args.agent in ("planner", "executor", "summarizer", "all"):
        patch_build_llm(args.gpu)

    profiling_ctx = None
    plan = None
    optimized_source = None

    if args.agent in ("profiler", "all"):
        profiling_ctx = test_profiler()

    if args.agent in ("planner", "all"):
        if profiling_ctx is None:
            # Load cached profiling context from output.csv
            from kernelopt.agents.profiler import build_profiling_context
            from kernelopt.functions.csv_parser import parse_ncu_csv
            csv_path = str(Path(__file__).parent.parent / "output.csv")
            if Path(csv_path).exists():
                print(f"\n  Using cached CSV: {csv_path}")
                sections, rules, launch = parse_ncu_csv(csv_path)
                profiling_ctx = build_profiling_context(sections, rules, launch,
                                                        str(Path(__file__).parent / "softmaxtri.py"))
            else:
                print("  No profiling context available. Run --agent profiler first.")
                sys.exit(1)
        plan = test_planner(profiling_ctx, args.gpu)

    if args.agent in ("executor", "all"):
        if plan is None:
            # Create a minimal test plan
            plan = {
                "plan_id": "test001",
                "direction": "use_exp2_instead_of_exp",
                "framework": "triton",
                "evidence": {
                    "ncu_rule": "test",
                    "ncu_rule_speedup_pct": 5.0,
                    "supporting_metrics": {},
                    "reasoning": "tl.math.exp2 is faster than tl.exp on GPU"
                },
                "step": {
                    "action": "replace",
                    "target": "tl.exp call",
                    "change": "Replace tl.exp(x) with tl.math.exp2(x * 1.4426950408889634) for faster computation",
                    "predicted_effect": "Reduced instruction latency"
                },
                "correctness_constraints": ["Output must remain numerically equivalent within 1e-4"],
                "confidence": "medium",
                "confidence_reason": "Direct instruction replacement",
            }
            print(f"\n  Using test plan: {plan['direction']}")
        optimized_source, _ = test_executor(plan, args.gpu)

    if args.agent in ("summarizer", "all"):
        if profiling_ctx is None:
            from kernelopt.agents.profiler import build_profiling_context
            from kernelopt.functions.csv_parser import parse_ncu_csv
            csv_path = str(Path(__file__).parent.parent / "output.csv")
            if Path(csv_path).exists():
                sections, rules, launch = parse_ncu_csv(csv_path)
                profiling_ctx = build_profiling_context(sections, rules, launch,
                                                        str(Path(__file__).parent / "softmaxtri.py"))
            else:
                print("  No profiling context available.")
                sys.exit(1)
        if plan is None:
            plan = {"plan_id": "test001", "direction": "test", "framework": "triton",
                    "step": {"action": "replace", "target": "exp", "change": "test", "predicted_effect": "test"}}
        if optimized_source is None:
            optimized_source = Path(Path(__file__).parent / "softmaxtri.py").read_text()
        test_summarizer(profiling_ctx, plan, optimized_source, args.gpu)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
