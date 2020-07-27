import random
import sys
import torch

sys.path.append("..")
from utils import CTCLoss

from time_utils import time_func

if getattr(sys.flags, "nogil", False) and sys.flags.nogil:
    print("Running without GIL")
else:
    print("Running with GIL")

torch.set_num_threads(1)
T = 250
L = 44
N = 80
B = int(sys.argv[1])
ITERATIONS = 100
inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True).cuda()
tgt = torch.randint(N - 2, (B, L)).split(1)
tgt = [t.tolist()[0] for t in tgt]

def func():
    inputs.grad = None
    op = CTCLoss(inputs, tgt, N - 1)
    op.backward()

time_func(func, name="asg fwd + bwd")
