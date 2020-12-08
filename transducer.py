"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn
import math
import numpy as np
import torch
import itertools


def make_scalar_graph(weight):
    scalar = gtn.Graph()
    scalar.add_node(True)
    scalar.add_node(False, True)
    scalar.add_arc(0, 1, 0, 0, weight)
    return scalar


def make_chain_graph(sequence):
    graph = gtn.Graph(False)
    graph.add_node(True)
    for i, s in enumerate(sequence):
        graph.add_node(False, i == (len(sequence) - 1))
        graph.add_arc(i, i + 1, s)
    return graph


def make_transitions_graph(ngram, num_tokens, calc_grad=False):
    transitions = gtn.Graph(calc_grad)
    transitions.add_node(True, ngram == 1)

    state_map = {(): 0}

    # first build transitions which include <s>:
    for n in range(1, ngram):
        for state in itertools.product(range(num_tokens), repeat=n):
            in_idx = state_map[state[:-1]]
            out_idx = transitions.add_node(False, ngram == 1)
            state_map[state] = out_idx
            transitions.add_arc(in_idx, out_idx, state[-1])

    for state in itertools.product(range(num_tokens), repeat=ngram):
        state_idx = state_map[state[:-1]]
        new_state_idx = state_map[state[1:]]
        # p(state[-1] | state[:-1])
        transitions.add_arc(state_idx, new_state_idx, state[-1])

    if ngram > 1:
        # build transitions which include </s>:
        end_idx = transitions.add_node(False, True)
        for in_idx in range(end_idx):
            transitions.add_arc(in_idx, end_idx, gtn.epsilon)

    return transitions


def make_lexicon_graph(word_pieces, graphemes_to_idx):
    """
    Constructs a graph which transduces letters to word pieces.
    """
    graph = gtn.Graph(False)
    graph.add_node(True, True)
    for i, wp in enumerate(word_pieces):
        prev = 0
        for l in wp[:-1]:
            n = graph.add_node()
            graph.add_arc(prev, n, graphemes_to_idx[l], gtn.epsilon)
            prev = n
        graph.add_arc(prev, 0, graphemes_to_idx[wp[-1]], i)
    graph.arc_sort()
    return graph


def make_token_graph(token_list, blank="none", allow_repeats=True):
    """
    Constructs a graph with all the individual
    token transition models.
    """
    if not allow_repeats and blank != "optional":
        raise ValueError("Must use blank='optional' if disallowing repeats.")

    ntoks = len(token_list)
    graph = gtn.Graph(False)

    # Creating nodes
    graph.add_node(True, True)
    for i in range(ntoks):
        # We can consume one or more consecutive
        # word pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        graph.add_node(False, blank != "forced")

    if blank != "none":
        graph.add_node()

    # Creating arcs
    if blank != "none":
        # blank index is assumed to be last (ntoks)
        graph.add_arc(0, ntoks + 1, ntoks, gtn.epsilon)
        graph.add_arc(ntoks + 1, 0, gtn.epsilon)

    for i in range(ntoks):
        graph.add_arc((ntoks + 1) if blank == "forced" else 0, i + 1, i)
        graph.add_arc(i + 1, i + 1, i, gtn.epsilon)

        if allow_repeats:
            if blank == "forced":
                # allow transition from token to blank only
                graph.add_arc(i + 1, ntoks + 1, ntoks, gtn.epsilon)
            else:
                # allow transition from token to blank and all other tokens
                graph.add_arc(i + 1, 0, gtn.epsilon)
        else:
            # allow transitions to blank and all other tokens except the same token
            graph.add_arc(i + 1, ntoks + 1, ntoks, gtn.epsilon)
            for j in range(ntoks):
                if i != j:
                    graph.add_arc(i + 1, j + 1, j, j)
    return graph


class Transducer(torch.nn.Module):
    """
    A generic transducer loss function.

    Args:
        tokens (list) : A list of iterable objects (e.g. strings, tuples, etc)
            representing the output tokens of the model (e.g. letters,
            word-pieces, words). For example ["a", "b", "ab", "ba", "aba"]
            could be a list of sub-word tokens.
        graphemes_to_idx (dict) : A dictionary mapping grapheme units (e.g.
            "a", "b", ..) to their corresponding integer index.
        ngram (int) : Order of the token-level transition model. If `ngram=0`
            then no transition model is used.
        blank (string) : Specifies the usage of blank token
            'none' - do not use blank token
            'optional' - allow an optional blank inbetween tokens
            'forced' - force a blank inbetween tokens (also referred to as garbage token)
        allow_repeats (boolean) : If false, then we don't allow paths with
            consecutive tokens in the alignment graph. This keeps the graph
            unambiguous in the sense that the same input cannot transduce to
            different outputs.
    """

    def __init__(
        self,
        tokens,
        graphemes_to_idx,
        ngram=0,
        transitions=None,
        blank="none",
        allow_repeats=True,
        reduction="none",
    ):
        super(Transducer, self).__init__()
        if blank not in ["optional", "forced", "none"]:
            raise ValueError(
                "Invalid value specificed for blank. Must be in ['optional', 'forced', 'none']"
            )
        self.tokens = make_token_graph(tokens, blank=blank, allow_repeats=allow_repeats)
        self.lexicon = make_lexicon_graph(tokens, graphemes_to_idx)
        self.ngram = ngram
        if ngram > 0 and transitions is not None:
            raise ValueError("Only one of ngram and transitions may be specified")
        if ngram > 0:
            transitions = make_transitions_graph(
                ngram, len(tokens) + int(blank != "none"), True
            )

        if transitions is not None:
            self.transitions = transitions
            self.transitions.arc_sort()
            self.transition_params = torch.nn.Parameter(
                torch.zeros(self.transitions.num_arcs())
            )
        else:
            self.transitions = None
            self.transition_params = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.transitions is None:
            inputs = torch.nn.functional.log_softmax(inputs, dim=2)
        self.tokens.arc_sort(True)
        return TransducerLoss(
            inputs,
            targets,
            self.tokens,
            self.lexicon,
            self.transition_params,
            self.transitions,
            self.reduction,
        )

    def viterbi(self, outputs):
        B, T, C = outputs.shape

        if self.transitions is not None:
            cpu_data = self.transition_params.cpu().contiguous()
            self.transitions.set_weights(cpu_data.data_ptr())
            self.transitions.calc_grad = False

        self.tokens.arc_sort()

        paths = [None] * B
        def process(b):
            emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            if self.transitions is not None:
                full_graph = gtn.intersect(emissions, self.transitions)
            else:
                full_graph = emissions

            # Find the best path and remove back-off arcs:
            path = gtn.remove(gtn.viterbi_path(full_graph))
            # Left compose the viterbi path with the "alignment to token"
            # transducer to get the outputs:
            path = gtn.compose(path, self.tokens)

            # When there are ambiguous paths (allow_repeats is true), we take
            # the shortest:
            path = gtn.viterbi_path(path)
            path = gtn.remove(gtn.project_output(path))
            paths[b] = path.labels_to_list()

        gtn.parallel_for(process, range(B))
        predictions = [torch.IntTensor(path) for path in paths]
        return predictions


class TransducerLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        targets,
        tokens,
        lexicon,
        transition_params=None,
        transitions=None,
        reduction="none",
    ):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B
        if transitions is not None:
            if transition_params is None:
                raise ValueError("Specified transitions, but not transition params.")
            cpu_data = transition_params.cpu().contiguous()
            transitions.set_weights(cpu_data.data_ptr())
            transitions.calc_grad = transition_params.requires_grad
            transitions.zero_grad()

        def process(b):
            # Create emissions graph:
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            cpu_data = inputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            target = make_chain_graph(targets[b])
            target.arc_sort(True)

            # Create token to grapheme decomposition graph
            tokens_target = gtn.remove(gtn.project_output(gtn.compose(target, lexicon)))
            tokens_target.arc_sort()

            # Create alignment graph:
            alignments = gtn.project_input(
                gtn.remove(gtn.compose(tokens, tokens_target))
            )
            alignments.arc_sort()

            # Add transition scores:
            if transitions is not None:
                alignments = gtn.intersect(transitions, alignments)
                alignments.arc_sort()

            loss = gtn.forward_score(gtn.intersect(emissions, alignments))

            # Normalize if needed:
            if transitions is not None:
                norm = gtn.forward_score(gtn.intersect(emissions, transitions))
                loss = gtn.subtract(loss, norm)

            losses[b] = gtn.negate(loss)

            # Save for backward:
            if emissions.calc_grad:
                emissions_graphs[b] = emissions

        gtn.parallel_for(process, range(B))

        ctx.graphs = (losses, emissions_graphs, transitions)
        ctx.input_shape = inputs.shape

        # Optionally reduce by target length:
        if reduction == "mean":
            scales = [(1 / len(t) if len(t) > 0 else 1.0) for t in targets]
        else:
            scales = [1.0] * B
        ctx.scales = scales

        loss = torch.tensor([l.item() * s for l, s in zip(losses, scales)])
        return torch.mean(loss.to(inputs.device))

    @staticmethod
    def backward(ctx, grad_output):
        losses, emissions_graphs, transitions = ctx.graphs
        scales = ctx.scales
        B, T, C = ctx.input_shape
        calc_emissions = ctx.needs_input_grad[0]
        input_grad = torch.empty((B, T, C)) if calc_emissions else None

        def process(b):
            scale = make_scalar_graph(scales[b])
            gtn.backward(losses[b], scale)
            emissions = emissions_graphs[b]
            if calc_emissions:
                grad = emissions.grad().weights_to_numpy()
                input_grad[b] = torch.tensor(grad).view(1, T, C)

        gtn.parallel_for(process, range(B))

        if calc_emissions:
            input_grad = input_grad.to(grad_output.device)
            input_grad *= grad_output / B

        if ctx.needs_input_grad[4]:
            grad = transitions.grad().weights_to_numpy()
            transition_grad = torch.tensor(grad).to(grad_output.device)
            transition_grad *= grad_output / B
        else:
            transition_grad = None

        return (
            input_grad,
            None,  # target
            None,  # tokens
            None,  # lex
            transition_grad,  # transition params
            None,  # transitions graph
            None,
        )


def make_kernel_graph(x, blank_idx, blank_optional, spike=False, calc_grad=False):
    g = gtn.Graph(calc_grad)
    g.add_node(True, len(x) == 0)  # start in blank
    g.add_arc(0, 0, blank_idx)
    for i, c in enumerate(x):
        g.add_node(False, blank_optional and (i + 1) == len(x))
        g.add_node(False, (i + 1) == len(x))
        g.add_arc(2 * i, 2 * i + 1, c)
        if not spike:
            g.add_arc(2 * i + 1, 2 * i + 1, c)
        g.add_arc(2 * i + 1, 2 * i + 2, blank_idx)
        g.add_arc(2 * i + 2, 2 * i + 2, blank_idx)
        if i > 0 and blank_optional and x[i - 1] != c:
            g.add_arc(2 * i - 1, 2 * i + 1, c)
    g.arc_sort(True)
    g.arc_sort()
    return g


class ConvTransduce1D(torch.nn.Module):
    """
    A 1D convolutional transducer layer.
    """

    def __init__(
        self,
        lexicon,
        kernel_size,
        stride,
        blank_idx,
        blank_optional=True,
        learn_params=False,
        scale="none",
        normalize="none",
        viterbi=False,
        spike=False,
    ):
        """
        Args:
            learn_params: If True, learn the kernel parameters
            scale ("none"): Scale the scores as a function of the
                kernel size. Can be any of "none", "sqrt", "linear".
            normalize ("none"): Normalize output scores. Can be any
                of "none", "pre", "post".
            viterbi: If True use the viterbi score intead of the
                forward score as output.
        """
        super(ConvTransduce1D, self).__init__()
        self.normalize = normalize
        self.viterbi = viterbi
        if scale == "none":
            self.scale = 1.0
        elif scale == "sqrt":
            self.scale = math.sqrt(kernel_size)
        elif scale == "linear":
            self.scale = kernel_size
        else:
            raise ValueError(f"Unknown scale {scale}")
        if normalize not in ["none", "pre", "post"]:
            raise ValueError(f"Unknown normalization {normalize}")

        # The lexicon consists of a list of iterable items. Each iterable
        # represents the indices of the subtokens (inputs from the previous
        # layer) which make up a token (output to the next layer).
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0, "Use an odd kernel size for easy padding."
        self.stride = stride

        def size_with_rep(token):
            reps = sum(t1 == t2 for t1, t2 in zip(token[:-1], token[1:]))
            return len(token) + reps

        min_kernel_size = max(size_with_rep(l) for l in lexicon)
        if kernel_size < min_kernel_size:
            raise ValueError(f"Kernel size needed of at least {min_kernel_size}.")
        self.kernels = [
            make_kernel_graph(l, blank_idx, blank_optional, spike=spike)
            for l in lexicon
        ]

        num_arcs = sum(k.num_arcs() for k in self.kernels)
        self.kernel_params = None
        if learn_params:
            self.kernel_params = torch.nn.Parameter(torch.zeros(num_arcs))

    def forward(self, inputs):
        # inputs are of shape [B, T, C]
        pad = self.kernel_size // 2
        inputs = torch.nn.functional.pad(inputs, (0, 0, pad, pad))
        if self.normalize == "pre":
            inputs = torch.nn.functional.log_softmax(inputs, dim=2)
        outputs = ConvTransduce1DFunction.apply(
            inputs,
            self.kernels,
            self.kernel_size,
            self.stride,
            self.kernel_params,
            self.viterbi,
        )
        outputs = outputs / self.scale
        if self.normalize == "post":
            outputs = torch.nn.functional.softmax(outputs, dim=2)
        if self.normalize == "pre":
            outputs = outputs.exp()
        return outputs


CTX_GRAPHS = None


class ConvTransduce1DFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs, kernels, kernel_size, stride, kernel_params=None, viterbi=False
    ):
        B, T, C = inputs.shape
        if T < kernel_size:
            # Padding should be done outside of this function:
            raise ValueError(f"Input ({T}) too short for kernel ({kernel_size})")
        cpu_inputs = inputs.cpu()
        output_graphs = [[] for _ in range(B)]
        input_graphs = [[] for _ in range(B)]

        if kernel_params is not None:
            cpu_data = kernel_params.cpu().contiguous()
            s = 0
            for kernel in kernels:
                na = kernel.num_arcs()
                data_ptr = cpu_data[s : s + na].data_ptr()
                s += na
                kernel.set_weights(data_ptr)
                kernel.calc_grad = kernel_params.requires_grad
                kernel.zero_grad()

        def process(b):
            for t in range(0, T - kernel_size + 1, stride):
                input_graph = gtn.linear_graph(kernel_size, C, inputs.requires_grad)
                window = cpu_inputs[b, t : t + kernel_size, :].contiguous()
                input_graph.set_weights(window.data_ptr())
                if viterbi:
                    window_outputs = [
                        gtn.viterbi_score(gtn.intersect(input_graph, kernel))
                        for kernel in kernels
                    ]
                else:
                    window_outputs = [
                        gtn.forward_score(gtn.intersect(input_graph, kernel))
                        for kernel in kernels
                    ]
                output_graphs[b].append(window_outputs)

                # Save for backward:
                if input_graph.calc_grad:
                    input_graphs[b].append(input_graph)

        gtn.parallel_for(process, range(B))

        global CTX_GRAPHS
        CTX_GRAPHS = (output_graphs, input_graphs, kernels)
        ctx.input_shape = inputs.shape
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        outputs = [
            [[o.item() for o in window] for window in example]
            for example in output_graphs
        ]
        return torch.tensor(outputs).to(inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        output_graphs, input_graphs, kernels = CTX_GRAPHS
        B, T, C = ctx.input_shape
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        input_grad = torch.zeros((B, T, C))
        deltas = grad_output.cpu().numpy()

        def process(b):
            for t, window in enumerate(output_graphs[b]):
                for c, out in enumerate(window):
                    delta = make_scalar_graph(deltas[b, t, c])
                    gtn.backward(out, delta)
                grad = (
                    input_graphs[b][t]
                    .grad()
                    .weights_to_numpy()
                    .reshape(kernel_size, -1)
                )
                input_grad[b, t * stride : t * stride + kernel_size] += grad

        gtn.parallel_for(process, range(B))

        if ctx.needs_input_grad[4]:
            kernel_grads = [k.grad().weights_to_numpy() for k in kernels]
            kernel_grads = np.concatenate(kernel_grads)
            kernel_grads = torch.from_numpy(kernel_grads).to(grad_output.device)
        else:
            kernel_grads = None
        return (
            input_grad.to(grad_output.device),
            None,  # kernels
            None,  # kernel_size
            None,  # stride
            kernel_grads,
            None,  # viterbi
        )


TransducerLoss = TransducerLossFunction.apply
