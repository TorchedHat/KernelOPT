"""KernelOPT output: torch.compile baseline.

KernelOPT could not produce a faster kernel for this model.
This file wraps the original model with torch.compile.
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]

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
