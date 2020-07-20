from concurrent.futures import ThreadPoolExecutor
import gtn
import torch


def thread_init():
    torch.set_num_threads(1)


def make_chain_graph(sequence):
    graph = gtn.Graph(False)
    graph.add_node(True)
    for i, s in enumerate(sequence):
        graph.add_node(False, i == (len(sequence) - 1))
        graph.add_arc(i, i + 1, s)
    return graph


def make_lexicon_graph(word_pieces, graphemes_to_idx):
    """
    Constructs a graph which transudces letters to word pieces.
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
    graph.arc_sort(True)
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
        n_gram (int) : Order of the token-level transition model. If `n_gram=0`
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
            n_gram=0,
            blank=False,
            allow_repeats=True,
            reduction="none"):
        super(Transducer, self).__init__()
        self.tokens = make_token_graph(
            tokens, blank=blank, allow_repeats=allow_repeats)
        self.lexicon = make_lexicon_graph(tokens, graphemes_to_idx)
        if n_gram > 0:
            raise NotImplementedError("Transition graphs not yet implemented.")
        self.transitions = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.transitions is None:
            inputs = torch.nn.functional.log_softmax(inputs, dim=2)
        inputs = inputs.permute(1, 0, 2) # T x B X C ->  B x T x C
        return TransducerLoss(
            inputs,
            targets,
            self.tokens,
            self.lexicon,
            self.transitions,
            self.reduction)

    def viterbi(self, outputs):
        B, T, C = outputs.shape

        def process(b):
            prediction = []
            emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            if self.transitions is not None:
                full_graph = gtn.intersect(emissions, transitions)
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
    def forward(ctx, inputs, targets, tokens, lexicon, transitions=None, reduction="none"):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B
        transitions_graphs = [None] * B
        def process(b):
            # Create emissions graph:
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            cpu_data = inputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            target = make_chain_graph(targets[b])
            target.arc_sort(True)

            # Create token to grapheme decomposition graph
            tokens_target = gtn.remove(
                gtn.project_output(gtn.compose(target, lexicon)))
            tokens_target.arc_sort()

            # Create alignment graph:
            alignments = gtn.project_input(
                gtn.remove(gtn.compose(tokens, tokens_target)))
            alignments.arc_sort()

            num = gtn.forward_score(gtn.intersect(emissions, alignments))
            if transitions is not None:
                denom = gtn.forward_score(gtn.intersect(emissions, transitions))
                losses[b] = gtn.subtract(denom, num)
            else:
                losses[b] = gtn.negate(num)

            # Save for backward:
            emissions_graphs[b] = emissions
            transitions_graphs[b] = transitions

        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()
        ctx.graphs = (losses, emissions_graphs, transitions_graphs)
        ctx.input_shape = inputs.shape

        # Optionally reduce by target length:
        if reduction == "mean":
            scales = [(1 / len(t) if len(t) > 0 else 1.0) for t in targets]
        else:
            scales = [1.0] * B
        ctx.scales = scales

        loss = torch.tensor([l.item() * s for l, s in zip(losses, scales)])
        return torch.mean(loss.cuda() if inputs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        losses, emissions_graphs, transitions_graphs = ctx.graphs
        scales = ctx.scales
        B, T, C = ctx.input_shape
        calc_emissions = emissions_graphs[0].calc_grad()
        calc_transitions = transitions_graphs[0] is not None \
                and transitions_graphs[0].calc_grad()
        input_grad = transitions_grad = None
        if calc_emissions:
            input_grad = torch.empty((B, T, C))
        if calc_transitions:
            transitions_grad = torch.empty(transtions_graphs[0].num_arcs())
        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            transitions = transitions_graphs[b]
            if calc_emissions:
                grad = emissions.grad().weights_to_numpy()
                input_grad[b] = torch.tensor(grad).view(1, T, C) * scales[b]
                # TODO don't save emissions graph in forward if calc_emissions is False
            if calc_transitions:
                raise NotImplementedError("Transitions not implemented yet.")
            # TODO, clean-up emissions and transitions graphs?


        executor = ThreadPoolExecutor(max_workers=B, initializer=thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            f.result()

        if input_grad is not None:
            if grad_output.is_cuda:
                input_grad = input_grad.cuda()
            input_grad *= (grad_output / B)
        if transitions_grad is not None:
            if grad_output.is_cuda:
                transitions_grad = transitions_grad.cuda()
            transitions_grad *= (grad_output / B)

        return (
            input_grad,
            None, # target
            None, # tokens
            None, # lex
            transitions_grad,
            None,
        )


TransducerLoss = TransducerLossFunction.apply
