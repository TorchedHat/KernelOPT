# Test Fixtures

This directory contains test fixtures and sample data for the kernel optimization test suite.

## Directory Structure

```
fixtures/
├── kernels/           # Sample GPU kernels for testing
│   ├── softmaxtri.py  # Triton softmax kernel (primary test kernel)
│   └── simple_add.py  # Simple addition kernel (for basic tests)
│
└── data/              # Sample profiling data and outputs
    ├── sample_ncu_output.csv       # Simple synthetic NCU CSV for basic tests
    └── h200_sample_ncu_output.csv  # Real H200 NCU output (comprehensive)
```

## Usage

Fixtures are used by tests via pytest fixtures defined in `conftest.py`:

```python
@pytest.fixture
def test_kernel_path():
    """Path to the softmax test kernel."""
    return Path(__file__).parent / "softmaxtri.py"

@pytest.fixture
def sample_ncu_csv(temp_output_dir):
    """Path to sample NCU CSV file."""
    # Returns path to fixtures/data/sample_ncu_output.csv
```

## Adding New Fixtures

### New Test Kernels

Add new test kernels to `kernels/`:

```python
# kernels/my_new_kernel.py
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    ...
```

### New Sample Data

Add new sample profiling data to `data/`:

- NCU CSV files: `data/kernel_name_ncu.csv`
- JSON files: `data/sample_plan.json`
- Other data: `data/sample_whatever.ext`

## Test Kernels

### `softmaxtri.py`

- **Purpose**: Primary test kernel for profiling and optimization
- **Framework**: Triton
- **Operation**: Fused softmax (2D)
- **Usage**: Most agent tests, profiling tests, E2E tests
- **Features**:
  - Numerical stability (row max subtraction)
  - Configurable block size
  - Warmup and benchmarking functions

### `simple_add.py`

- **Purpose**: Simple kernel for basic correctness tests
- **Framework**: Triton
- **Operation**: Element-wise addition
- **Usage**: Quick correctness checks, parser tests
- **Features**:
  - Minimal complexity
  - Fast execution
  - Easy to verify correctness

## Sample Data

### `sample_ncu_output.csv`

- **Purpose**: Simple synthetic NCU profiling output for basic CSV parser tests
- **Kernel**: softmax_kernel
- **Metrics**: Duration, throughput metrics, launch config
- **Size**: 485 bytes (10 lines)
- **Usage**: Basic CSV parser tests, quick validation tests
- **Format**: Simplified NCU CSV format with essential metrics

### `h200_sample_ncu_output.csv`

- **Purpose**: Real H200 GPU profiling output for comprehensive testing
- **Source**: Actual NCU profiling run on NVIDIA H200 GPU
- **Kernel**: PyTorch CUDA kernel (distribution_elementwise_grid_stride_kernel)
- **GPU**: NVIDIA H200 (Compute Capability 9.0)
- **Metrics**: Complete NCU output with all sections and rules
- **Size**: 172 KB (212 lines)
- **Usage**: Comprehensive CSV parser tests, profiling agent tests, integration tests
- **Features**:
  - Real GPU Speed Of Light metrics
  - NCU rules (SOLBottleneck, PMSamplingData, etc.)
  - Multiple profiling sections
  - Realistic metric values from H200
- **Sections Included**:
  - GPU Speed Of Light Throughput
  - PM Sampling
  - Compute Workload Analysis
  - NCU Rules and Recommendations

## Best Practices

1. **Keep kernels simple** - Test kernels should be simple and focused
2. **Document expected behavior** - Add comments about what each kernel does
3. **Provide reference outputs** - Include expected outputs for correctness tests
4. **Version control** - Commit fixtures with tests
5. **Clean up** - Remove unused fixtures regularly
