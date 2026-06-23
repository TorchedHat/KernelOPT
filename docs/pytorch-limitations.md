# PyTorch Limitations Affecting kernelopt

Known torch.compile / Inductor limitations that affect the kernelopt pipeline. These are upstream issues — not fixable on our side.

## 1. `NoValidChoicesError` with mixed-precision fp16 linear + fp32 output

**Discovered:** 2026-05-20
**Torch version:** 2.13.0a0+git43ec629
**Reproduction:**

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 256)
    def forward(self, x):
        return self.linear(x.half()).float()
```

With `torch._inductor.config.max_autotune = True`, torch.compile raises:
```
torch._inductor.exc.InductorError: NoValidChoicesError
```
from `select_algorithm.py:do_autotuning` — the autotuner can't find a valid GEMM implementation for the fp16→fp32 cast path.

**Workaround:** Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)` instead of explicit `.half()` casts. BF16 autocast works correctly.

**Impact:** Kernels that use explicit fp16 casting with `max_autotune=True` cannot be synthesized via torch.compile. The pipeline falls back gracefully (synthesis failure → continue with original PyTorch file).

**Upstream issue:** TODO — file on pytorch/pytorch.

## 2. `_skip_one_hop_torch_function` AttributeError

**Discovered:** 2026-05-20
**Torch version:** 2.13.0a0+git43ec629
**Reproduction:**

```python
@torch.compile
def fn(x):
    return torch.softmax(x, dim=-1)
fn(torch.randn(8, 1024, device='cuda'))
```

When run directly (not via subprocess with `run_and_get_code`), torch.compile raises:
```
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: module 'torch._C' has no attribute '_skip_one_hop_torch_function'
```

**Workaround:** Use `torch._inductor.utils.run_and_get_code()` in a subprocess (which is what our synthesis pipeline already does). The error doesn't occur in the subprocess path.

**Impact:** Cannot use `torch.compile` directly in the main kernelopt process. All synthesis must go through subprocess isolation, which is the existing design.

## 3. FP4 `float4_e2m1fn_x2` dtype casting not implemented

**Discovered:** 2026-05-20
**Torch version:** 2.13.0a0+git43ec629
**Reproduction:**

```python
x = torch.randn(128, 128, device='cuda')
fp4 = x.to(torch.float4_e2m1fn_x2)
# Error: copy_() does not support casting Float4_e2m1fn_x2 to different types
```

**Impact:** SOL-ExecBench quantized kernels using FP4 dtypes (e.g., `kernel_007` — nvfp4 multimodal projection) cannot be synthesized. The `torch._scaled_mm` operator requires both inputs to be FP4/FP8, but standard tensor casting to FP4 is not implemented.

**Upstream issue:** TODO — file on pytorch/pytorch.
