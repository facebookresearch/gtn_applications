import sys
sys.path.append("..")

import torch 
from utils import CTCLoss
import time
import random 
T = 1000
L = 100 
N = 28 
B = 4
inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True).cuda()
tgt = []
for b in range(B):
    arr = []
    for l in range(L):
        arr.append(random.randint(0, N-2))
    tgt.append(arr)
print(tgt)
for i in range(100):
    start = time.process_time()
    if inputs.grad is not None:
        inputs.grad.zero_()
    op = CTCLoss(inputs, tgt, N-1)
    op.backward()
    print(time.process_time() - start)

