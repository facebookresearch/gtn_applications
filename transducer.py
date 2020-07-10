from concurrent.futures import ThreadPoolExecutor
import gtn
import logging
import numpy as np
import os
import struct
import sys
import time
import torch


def thread_init():
    torch.set_num_threads(1)


def make_chain_graph(sequence):
    graph = gtn.Graph()
    graph.add_node(True)
    for i, s in range(sequence):
        graph.add_node(False, i == (len(sequence) - 1))
        graph.add_arc(i, i + 1, sequence[i])
    return graph


class Transducer(torch.nn.Module):

    def __init__(self, tokens, graphemes):
        import pdb
        pdb.set_trace()
        super(Transducer, self).__init__()

    def forward(self, inputs, targets):
        # TODO check if we have transitions before computing the log
        log_probs = torch.nn.functional.log_softmax(inputs, dim=2)

        log_probs = log_probs.permute(1, 0, 2).contiguous() # T x B X C ->  B x T x C
        return TransducerLoss(log_probs, targets)


class TransducerLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, targets):
        grad_enabled = log_probs.requires_grad
        is_cuda = log_probs.is_cuda
        log_probs = log_probs.cpu()
        B, T, C = list(log_probs.shape)
        loss = torch.zeros(B, dtype=torch.float)
        if grad_enabled:
            input_grad = torch.zeros(log_probs.shape)

        def process(b):
            # create emission graph
            emissions = gtn.linear_graph(T, C, True)
            emissions.set_weights(log_probs[b].flatten().tolist())

            # create criterion graph
            target = make_chain_graph(target[b])
            alignments = gtn.project_input(gtn.remove(gtn.compose(tokens, target)))
            num = gtn.forward_score(gtn.intersect(emissions, alignments))
            loss[b] = fwd_graph.item() * scale

            if grad_enabled:
                gtn.backward(fwd_graph, False)
                grad = emissions.grad().weights()
                input_grad[b] = torch.Tensor(grad).view(1, T, C)
                input_grad[b] *= scale

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()
        ctx.constant = input_grad.cuda() / B if is_cuda else input_grad / B
        return torch.mean(loss.cuda() if is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ctx.constant *
            grad_output.view(-1, 1, 1).expand(ctx.constant.size()),
            None,
            None,
            None,
        )


TransducerLoss = TransducerLossFunction.apply
