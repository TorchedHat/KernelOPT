import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Constants
SEQ_LEN = 16384
BATCH = 2
EMBED_DIM = 128
NUM_HEADS = 8
HEAD_DIM = 16  # 128/8=16... wait graph shows [16384,8,32] so head_dim=32, num_heads=8

Let me re-examine: embed_dim=128, num_heads shown in graph=8, head_dim=32. But 8*32=256!=128. Actually looking again at the graph: `view.default -> shape=[16384, 8, 32]` - that's 8 heads * 32 dim = 256 != 128. Hmm, but the in_proj produces 384 = 3*128, split into Q,K,V each 128. Then each is viewed as [16384, 8, 16]? No the graph says 32...

Wait: 128/4=32, so 4 heads with head_dim=32. But graph shows shape=[16384, 8, 32]... 8*32=256!=128. Let me look at bmm output: [8, 16384, 16384] and [8, 16384, 32]. So 8 heads. But 8*32=256 and embed_dim=128... Unless the batch dimension 2 is being folded: 2 batches * 4 heads = 8. Yes! That's it. The bmm treats batch*heads as the batch dimension.

So: 4 heads, head_dim=32, batch=2, so effective batch for bmm = 2*4=8.

import torch
import triton
import triton.language as tl