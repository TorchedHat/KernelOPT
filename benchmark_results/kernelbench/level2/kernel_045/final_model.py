"""KernelOPT output: torch.compile baseline.

KernelOPT could not produce a faster kernel for this model.
This file wraps the original model with torch.compile.
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.logsumexp(x, dim=1)  # compute LogSumExp over features per sample
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]


# ── torch.compile baseline ──
import torch as _torch
_torch.manual_seed(42)
_torch.cuda.manual_seed(42)
_init = get_init_inputs() if "get_init_inputs" in dir() else []
_model = Model(*_init).cuda().eval()
_compiled = _torch.compile(_model)
# Warmup
with _torch.no_grad():
    _compiled(*[t.cuda() if isinstance(t, _torch.Tensor) else t for t in get_inputs()])

def kernel_function(*inputs):
    with _torch.no_grad():
        return _compiled(*inputs)
