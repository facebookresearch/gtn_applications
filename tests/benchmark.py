import random
import sys
import torch
import time

sys.path.append("..")
from utils import CTCLoss

if hasattr(sys.flags, 'nogil') and sys.flags.nogil:
    print("Running without GIL")
else:
    print("Running with GIL")

torch.set_num_threads(1)
T = 1000
L = 100
N = 28
B = int(sys.argv[1])
ITERATIONS = 100
inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True).cuda()
tgt = []
for b in range(B):
    arr = []
    for l in range(L):
        arr.append(random.randint(0, N - 2))
    tgt.append(arr)

#warmup
for i in range(5):
    if inputs.grad is not None:
        inputs.grad.zero_()
    op = CTCLoss(inputs, tgt, N - 1)
    op.backward()

start = time.perf_counter()
for i in range(ITERATIONS):
    if inputs.grad is not None:
        inputs.grad.zero_()
    op = CTCLoss(inputs, tgt, N - 1)
    op.backward()
print("Took", (time.perf_counter() - start) * 1000 / ITERATIONS, "ms")
