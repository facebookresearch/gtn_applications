import random
import sys
import torch
import time

sys.path.append("..")
from utils import ASGLoss

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
transitions = torch.randn(N + 1, N, dtype=torch.float, requires_grad=True).cuda()
tgt = torch.randint(N - 2, (B, L)).split(1)
tgt = [t.tolist()[0] for t in tgt]

# warmup
for i in range(5):
    inputs.grad = None
    transitions.grad = None
    op = ASGLoss(inputs, transitions, tgt)
    op.backward()

start = time.perf_counter()
for i in range(ITERATIONS):
    inputs.grad = None
    transitions.grad = None
    op = ASGLoss(inputs, transitions, tgt)
    op.backward()
print("Took", (time.perf_counter() - start) * 1000 / ITERATIONS, "ms")
