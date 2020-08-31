import random
import sys
import torch

sys.path.append("..")
import transducer

from time_utils import time_func


def word_decompositions():
    tokens_path = "/checkpoint/awni/data/iamdb/word_pieces_tokens_1000.txt"
    with open(tokens_path, "r") as fid:
        tokens = sorted([l.strip() for l in fid])
    graphemes = sorted(set(c for t in tokens for c in t))
    graphemes_to_index = {t: i for i, t in enumerate(graphemes)}

    N = len(tokens) + 1
    T = 100
    L = 15
    B = 1
    if len(sys.argv) > 1:
        B = int(sys.argv[1])

    inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True).cuda()

    targets = []
    for b in range(B):
        pieces = (random.choice(tokens) for l in range(L))
        target = [graphemes_to_index[l] for wp in pieces for l in wp]
        targets.append(torch.tensor(target))

    crit = transducer.Transducer(
        tokens, graphemes_to_index, blank="optional", allow_repeats=False, reduction="mean"
    )

    def fwd_bwd():
        loss = crit(inputs, targets)
        loss.backward()
    time_func(fwd_bwd, 20, "word decomps fwd + bwd")

    def viterbi():
        crit.viterbi(inputs)
    time_func(viterbi, 20, "word decomps viterbi")


def ngram_ctc():
    N = 81
    T = 250
    L = 44
    B = 1
    if len(sys.argv) > 1:
        B = int(sys.argv[1])

    tokens = [(i,) for i in range(N)]
    graphemes_to_index = {i : i for i in range(N)}

    ITERATIONS = 20
    inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True)

    targets = [tgt.squeeze()
        for tgt in torch.randint(N, size=(B, L)).split(1)]

    for ngram in [0, 1, 2]:
        crit = transducer.Transducer(
            tokens, graphemes_to_index,
            ngram=ngram, blank="optional",
            allow_repeats=False, reduction="mean"
        )
        def fwd_bwd():
            loss = crit(inputs, targets)
            loss.backward()
        time_func(
            fwd_bwd, iterations=20, name=f"ctc fwd + bwd, ngram={ngram}")
        def viterbi():
            crit.viterbi(inputs)
        time_func(
            viterbi, iterations=20, name=f"ctc viterbi, ngram={ngram}")


def ngram_asg():
    N = 81
    T = 250
    L = 44
    B = 1
    if len(sys.argv) > 1:
        B = int(sys.argv[1])

    tokens = [(i,) for i in range(N)]
    graphemes_to_index = {i : i for i in range(N)}

    ITERATIONS = 20
    inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True)

    targets = [tgt.squeeze()
        for tgt in torch.randint(N, size=(B, L)).split(1)]

    for ngram in [0, 1, 2]:
        crit = transducer.Transducer(
            tokens, graphemes_to_index, ngram=ngram, reduction="mean"
        )
        def fwd_bwd():
            loss = crit(inputs, targets)
            loss.backward()
        time_func(
            fwd_bwd, iterations=20, name=f"asg fwd + bwd, ngram={ngram}")
        def viterbi():
            crit.viterbi(inputs)
        time_func(
            viterbi, iterations=20, name=f"asg viterbi, ngram={ngram}")


if __name__ == "__main__":
    if getattr(sys.flags, "nogil", False) and sys.flags.nogil:
        print("Running without GIL")
    else:
        print("Running with GIL")
    torch.set_num_threads(1)
    word_decompositions()
    ngram_ctc()
    ngram_asg()
