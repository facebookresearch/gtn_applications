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


def make_token_graph(token_list, blank=False):
    """
    Constructs a graph with all the individual
    token transition models.
    """
    graph = gtn.Graph(False)
    graph.add_node(True)
    for i, wp in enumerate(token_list):
        # We can consume one or more consecutive
        # word pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        #graph = gtn.Graph(False)
        graph.add_node(False, True)
        graph.add_arc(0, i + 1, i)
        graph.add_arc(i + 1, i + 1, i, gtn.epsilon)
        graph.add_arc(i + 1, 0, gtn.epsilon)
    if blank:
        i = len(token_list)
        graph.add_node(False, True)
        graph.add_arc(0, i + 1, i, gtn.epsilon)
        graph.add_arc(i + 1, 0, gtn.epsilon)
    return graph


class Transducer(torch.nn.Module):

    def __init__(self, tokens, graphemes_to_idx, n_gram=0, blank=False):
        super(Transducer, self).__init__()
        self.tokens = make_token_graph(tokens, blank=blank)
        self.lexicon = make_lexicon_graph(tokens, graphemes_to_idx)
        if n_gram > 0:
            raise NotImplementedError("Transition graphs not yet implemented.")
        self.transitions = None

    def forward(self, inputs, targets):
        if self.transitions is None:
            inputs = torch.nn.functional.log_softmax(inputs, dim=2)
        inputs = inputs.permute(1, 0, 2).contiguous() # T x B X C ->  B x T x C
        return TransducerLoss(inputs, targets, self.tokens, self.lexicon, self.transitions)


class TransducerLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, tokens, lexicon, transitions=None):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B
        transitions_graphs = [None] * B

        def process(b):
            # Create emissions graph:
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            # TODO, we ought to use data_ptr here and avoid conversions to/from python lists
            emissions.set_weights(inputs[b].cpu().flatten().tolist())
            target = make_chain_graph(targets[b])

            # Create token to grapheme decomposition graph
            tokens_target = gtn.remove(
                gtn.project_output(gtn.compose(target, lexicon)))

            # Create alignment graph:
            alignments = gtn.project_input(
                gtn.remove(gtn.compose(tokens, tokens_target)))

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
        loss = torch.tensor([l.item() for l in losses])
        return torch.mean(loss.cuda() if inputs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        # We need another multiproc fn.
        # We call loss backward and extract gradients w.r.t. the emission and transition
        losses, emissions_graphs, transitions_graphs = ctx.graphs
        B = len(emissions_graphs)
        T = emissions_graphs[0].num_nodes() - 1
        C = emissions_graphs[0].num_arcs() // T
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
                grad = emissions.grad().weights()
                input_grad[b] = torch.tensor(grad).view(1, T, C)
            if calc_transitions:
                raise NotImplementedError("Transitions not implemented yet.")

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
        )


TransducerLoss = TransducerLossFunction.apply
