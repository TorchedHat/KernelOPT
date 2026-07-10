"""KernelOPT output: torch.compile baseline.

KernelOPT could not produce a faster kernel for this model.
This file wraps the original model with torch.compile.
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]

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
