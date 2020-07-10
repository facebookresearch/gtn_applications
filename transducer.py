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
    graph = gtn.Graph(False)
    graph.add_node(True)
    for i, s in range(sequence):
        graph.add_node(False, i == (len(sequence) - 1))
        graph.add_arc(i, i + 1, sequence[i])
    return graph


def make_lexicon_graph(word_pieces, graphemes_to_idx):
    """
    Constructs a graph which transudces letters to word pieces.
    """
    # TODO, consider direct construction as it could be more efficient
    lex = []
    for i, wp in enumerate(word_pieces):
        graph = gtn.Graph(False)
        graph.add_node(True)
        for e, l in enumerate(wp):
            if e == len(wp) - 1:
                graph.add_node(False, True)
                graph.add_arc(e, e + 1, graphemes_to_idx[l], i)
            else:
                graph.add_node()
                graph.add_arc(e, e + 1, graphemes_to_idx[l], gtn.epsilon)
        lex.append(graph)
    return gtn.closure(gtn.sum(lex))


def make_token_graph(token_list):
    """
    Constructs a graph with all the individual
    token transition models.
    """
    # TODO, consider direct construction as it could be more efficient
    tokens = []
    for i, wp in enumerate(token_list):
        # We can consume one or more consecutive
        # word pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        graph = gtn.Graph(False)
        graph.add_node(True)
        graph.add_node(False, True)
        graph.add_arc(0, 1, i, i)
        graph.add_arc(1, 1, i, gtn.epsilon)
        tokens.append(graph)
    return gtn.closure(gtn.sum(tokens))


class Transducer(torch.nn.Module):

    def __init__(self, tokens, graphemes_to_idx, n_gram=0, blank=False):
        super(Transducer, self).__init__()
        self.tokens = make_token_graph(tokens)
        self.lexicon = make_lexicon_graph(tokens, graphemes_to_id)
        if n_gram > 0:
            raise NotImplementedError("Transition graphs not yet implemented.")
        if blank:
            raise NotImplementedError("Blank label not yet supported.")
        self.transitions = None

    def forward(self, inputs, targets):
        if transitions is None:
            inputs = torch.nn.functional.log_softmax(inputs, dim=2)
        inputs = inputs.permute(1, 0, 2).contiguous() # T x B X C ->  B x T x C
        return TransducerLoss(inputs, targets, self.tokens, self.lexicon, self.transitions)


class TransducerLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, tokens, lexicon, transitions=None):
        B, T, C = inputs.shape
        loss = torch.zeros(B, dtype=torch.float)

        def process(b):
            # Create emissions graph:
            emissions = gtn.linear_graph(T, C, True)
            emissions.set_weights(inputs[b].cpu().data_ptr)

            # Create alignment graph:
            target = make_chain_graph(target[b])
            alignments = gtn.project_input(gtn.remove(gtn.compose(tokens, target)))
            num = gtn.forward_score(gtn.intersect(emissions, alignments))
            if transitions is not None:
                denom = gtn.forward_score(gtn.intersect(emissions, transitions))
                loss[b] = gtn.subtract(denom, num).item()
            else:
                loss[b] = gtn.negate(num).item()

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()
        return torch.mean(loss.cuda() if inputs.is_cuda() else loss)

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
