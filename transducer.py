from concurrent.futures import ThreadPoolExecutor
import gtn
import torch
import itertools


def thread_init():
    torch.set_num_threads(1)


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
    transitions.add_node(True, True)

    # first build transitions which include <s>:
    for n in range(1, ngram):
        for state in itertools.product(range(num_tokens), repeat=n):
            in_idx = sum((s + 1) * (num_tokens**e) for e, s in enumerate(state[:-1]))
            out_idx = transitions.add_node(False, True)
            transitions.add_arc(in_idx, out_idx, state[-1])

    for state in itertools.product(range(num_tokens), repeat=ngram):
        state_idx = sum((s + 1) * (num_tokens**e) for e, s in enumerate(state[:-1]))
        new_state_idx = sum((s + 1) * (num_tokens**e) for e, s in enumerate(state[1:]))
        # p(state[-1] | state[:-1])
        transitions.add_arc(state_idx, new_state_idx, state[-1])
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


def make_token_graph(token_list, blank=False, allow_repeats=True):
    """
    Constructs a graph with all the individual
    token transition models.
    """
    ntoks = len(token_list)
    graph = gtn.Graph(False)
    graph.add_node(True, True)
    for i in range(ntoks):
        # We can consume one or more consecutive
        # word pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        graph.add_node(False, True)
        graph.add_arc(0, i + 1, i)
        graph.add_arc(i + 1, i + 1, i, gtn.epsilon)
        if allow_repeats:
            graph.add_arc(i + 1, 0, gtn.epsilon)

    if blank:
        graph.add_node()
        # blank index is assumed to be last (ntoks)
        graph.add_arc(0, ntoks + 1, ntoks, gtn.epsilon)
        graph.add_arc(ntoks + 1, 0, gtn.epsilon)

    if not allow_repeats:
        if not blank:
            raise ValueError("Must use blank if disallowing repeats.")
        # For each token, allow a transition on blank or a transition on all
        # other tokens.
        for i in range(ntoks):
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
        blank (boolean) : Toggle the use of an optional blank inbetween tokens.
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
        blank=False,
        allow_repeats=True,
        reduction="none",
    ):
        super(Transducer, self).__init__()
        self.tokens = make_token_graph(tokens, blank=blank, allow_repeats=allow_repeats)
        self.lexicon = make_lexicon_graph(tokens, graphemes_to_idx)
        self.ngram = ngram
        if ngram > 0:
            self.transitions = make_transitions_graph(ngram, len(tokens) + blank, True)
            self.transitions.arc_sort()
            self.transition_params = torch.nn.Parameter(
                torch.zeros(self.transitions.num_arcs()))
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
            self.reduction)

    def viterbi(self, outputs):
        B, T, C = outputs.shape

        if self.transitions is not None:
            cpu_data = self.transition_params.cpu().contiguous()
            self.transitions.set_weights(cpu_data.data_ptr())
            self.transitions.calc_grad = False

        self.tokens.arc_sort()

        def process(b):
            emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            if self.transitions is not None:
                full_graph = gtn.intersect(emissions, self.transitions)
            else:
                full_graph = emissions
            # Left compose the viterbi path with the "alignment to token"
            # transducer to get the outputs:
            path = gtn.compose(gtn.viterbi_path(full_graph), self.tokens)

            # When there are ambiguous paths (allow_repeats is true), we take
            # the shortest:
            path = gtn.viterbi_path(path)
            path = gtn.remove(gtn.project_output(path))
            return path.labels_to_list()

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        predictions = [torch.IntTensor(f.result()) for f in futures]
        executor.shutdown()
        return predictions


class TransducerLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, inputs, targets, tokens, lexicon,
            transition_params=None, transitions=None, reduction="none"):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B
        if transitions is not None:
            if transition_params is None:
                raise ValueError(
                    "Specified transitions, but not transition params.")
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

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()
        executor.shutdown()

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

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()
        executor.shutdown()

        if calc_emissions:
            input_grad = input_grad.to(grad_output.device)
            input_grad *= (grad_output / B)

        if ctx.needs_input_grad[4]:
            grad = transitions.grad().weights_to_numpy()
            transition_grad = torch.tensor(grad).to(grad_output.device)
            transition_grad *= (grad_output / B)
        else:
            transition_grad = None

        return (
            input_grad,
            None, # target
            None, # tokens
            None, # lex
            transition_grad, # transition params
            None, # transitions graph
            None,
        )


TransducerLoss = TransducerLossFunction.apply
