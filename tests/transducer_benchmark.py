import random
import sys
import torch
import time

sys.path.append("..")
import transducer

if hasattr(sys.flags, "nogil") and sys.flags.nogil:
    print("Running without GIL")
else:
    print("Running with GIL")

torch.set_num_threads(1)
tokens_path = "/checkpoint/awni/data/iamdb/word_pieces_tokens_1000.txt"
with open(tokens_path, 'r') as fid:
    tokens = sorted([l.strip() for l in fid])
graphemes = sorted(set(c for t in tokens for c in t))
graphemes_to_index = { t : i for i, t in enumerate(graphemes)}

N = len(tokens) + 1
T = 100
L = 15
B = 1
if len(sys.argv) > 1:
    B = int(sys.argv[1])

ITERATIONS = 20
inputs = torch.randn(T, B, N, dtype=torch.float, requires_grad=True).cuda()

targets = []
for b in range(B):
    pieces = (random.choice(tokens) for l in range(L))
    target = [graphemes_to_index[l] for wp in pieces for l in wp]
    targets.append(torch.tensor(target))

crit = transducer.Transducer(
    tokens,
    graphemes_to_index,
    blank=True,
    allow_repeats=False,
    reduction="mean")

# warmup:
for i in range(5):
    loss = crit(inputs, targets)
    loss.backward()

start = time.perf_counter()

for i in range(ITERATIONS):
    loss  = crit(inputs, targets)
    loss.backward()

print("Forward+Backward took {:.3f} ms".format((time.perf_counter() - start) * 1000 / ITERATIONS))

# warmup:
for i in range(5):
    viterbi_path = crit.viterbi(inputs)

start = time.perf_counter()

for i in range(ITERATIONS):
    viterbi_path = crit.viterbi(inputs)

print("Viterbi took {:.3f} ms".format((time.perf_counter() - start) * 1000 / ITERATIONS))
