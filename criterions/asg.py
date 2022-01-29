"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import gtn
import itertools


def pack_replabels(tokens, num_replabels):
    if all(isinstance(t, list) for t in tokens):
        return [pack_replabels(t, num_replabels) for t in tokens]
    assert isinstance(tokens, list)
    new_tokens = []
    L = len(tokens)
    num = 0
    prev_token = -1
    for token in tokens:
        if token == prev_token and num < num_replabels:
            num += 1
        else:
            if num > 0:
                new_tokens.append(num - 1)
                num = 0
            new_tokens.append(token + num_replabels)
            prev_token = token
    if num > 0:
        new_tokens.append(num - 1)
    return new_tokens


def unpack_replabels(tokens, num_replabels):
    if all(isinstance(t, list) for t in tokens):
        return [unpack_replabels(t, num_replabels) for t in tokens]
    assert isinstance(tokens, list)
    new_tokens = []
    prev_token = -1
    for token in tokens:
        if token >= num_replabels:
            new_tokens.append(token - num_replabels)
            prev_token = token
        elif prev_token != -1:
            for i in range(token + 1):
                new_tokens.append(prev_token - num_replabels)
            prev_token = -1
    return new_tokens


class ASGLossFunction(torch.autograd.Function):
    @staticmethod
    def create_transitions_graph(transitions, calc_grad=False):
        num_classes = transitions.shape[1]
        assert transitions.shape == (num_classes + 1, num_classes)
        g_transitions = gtn.Graph(calc_grad)
        g_transitions.add_node(True)
        for i in range(1, num_classes + 1):
            g_transitions.add_node(False, True)
            g_transitions.add_arc(0, i, i - 1)  #  p(i | <s>)
        for i in range(num_classes):
            for j in range(num_classes):
                g_transitions.add_arc(j + 1, i + 1, i)  # p(i | j)
        cpu_data = transitions.cpu().contiguous()
        g_transitions.set_weights(cpu_data.data_ptr())
        g_transitions.mark_arc_sorted(False)
        g_transitions.mark_arc_sorted(True)
        return g_transitions

    @staticmethod
    def create_force_align_graph(target):
        g_fal = gtn.Graph(False)
        L = len(target)
        g_fal.add_node(True)
        for l in range(1, L + 1):
            g_fal.add_node(False, l == L)
            g_fal.add_arc(l - 1, l, target[l - 1])
            g_fal.add_arc(l, l, target[l - 1])
        g_fal.arc_sort(True)
        return g_fal

    @staticmethod
    def forward(ctx, inputs, transitions, targets, reduction="none"):
        B, T, C = inputs.shape
        losses = [None] * B
        scales = [None] * B
        emissions_graphs = [None] * B
        transitions_graphs = [None] * B

        calc_trans_grad = transitions.requires_grad
        transitions = transitions.cpu()  # avoid multiple cuda -> cpu copies

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            cpu_data = inputs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create transition graph
            g_transitions = ASGLossFunction.create_transitions_graph(
                transitions, calc_trans_grad
            )

            # create force align criterion graph
            g_fal = ASGLossFunction.create_force_align_graph(targets[b])

            # compose the graphs
            g_fal_fwd = gtn.forward_score(
                gtn.intersect(gtn.intersect(g_fal, g_transitions), g_emissions)
            )
            g_fcc_fwd = gtn.forward_score(gtn.intersect(g_emissions, g_transitions))
            g_loss = gtn.subtract(g_fcc_fwd, g_fal_fwd)
            scale = 1.0
            if reduction == "mean":
                L = len(targets[b])
                scale = 1.0 / L if L > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid value for reduction '" + str(reduction) + "'")

            # Save for backward:
            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions
            transitions_graphs[b] = g_transitions

        gtn.parallel_for(process, range(B))

        ctx.auxiliary_data = (
            losses,
            scales,
            emissions_graphs,
            transitions_graphs,
            inputs.shape,
        )
        loss = torch.tensor([losses[b].item() * scales[b] for b in range(B)])
        return torch.mean(loss.cuda() if inputs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        (
            losses,
            scales,
            emissions_graphs,
            transitions_graphs,
            in_shape,
        ) = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = transitions_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = torch.empty((B, T, C))
        if ctx.needs_input_grad[1]:
            transitions_grad = torch.empty((B, C + 1, C))

        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            transitions = transitions_graphs[b]
            if input_grad is not None:
                grad = emissions.grad().weights_to_numpy()
                input_grad[b] = torch.from_numpy(grad).view(1, T, C) * scales[b]
            if transitions_grad is not None:
                grad = transitions.grad().weights_to_numpy()
                transitions_grad[b] = (
                    torch.from_numpy(grad).view(1, C + 1, C) * scales[b]
                )

        gtn.parallel_for(process, range(B))
        if input_grad is not None:
            if grad_output.is_cuda:
                input_grad = input_grad.cuda()
            input_grad *= grad_output / B
        if transitions_grad is not None:
            if grad_output.is_cuda:
                transitions_grad = transitions_grad.cuda()

            transitions_grad = torch.mean(transitions_grad, 0) * grad_output
        return (
            input_grad,
            transitions_grad,
            None,  # target
            None,  # reduction
        )


ASGLoss = ASGLossFunction.apply


class ASG(torch.nn.Module):
    def __init__(self, num_classes, num_replabels=1, use_garbage=True):
        super(ASG, self).__init__()
        self.num_classes = num_classes
        self.num_replabels = num_replabels
        assert self.num_replabels > 0
        self.garbage_idx = (num_classes + num_replabels) if use_garbage else None
        self.N = num_classes + num_replabels + int(use_garbage)
        self.transitions = torch.nn.Parameter(torch.zeros(self.N + 1, self.N))

    def forward(self, inputs, targets):
        targets = [pack_replabels(t.tolist(), self.num_replabels) for t in targets]
        if self.garbage_idx is not None:
            # add a garbage token between each target label
            for idx in range(len(targets)):
                prev_tgt = targets[idx]
                targets[idx] = [self.garbage_idx] * (len(prev_tgt) * 2 + 1)
                targets[idx][1::2] = prev_tgt
        return ASGLoss(inputs, self.transitions, targets, "mean")

    def viterbi(self, outputs):
        B, T, C = outputs.shape
        assert C == self.N, "Wrong number of classes in output."

        predictions = [None] * B

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create transition graph
            g_transitions = ASGLossFunction.create_transitions_graph(self.transitions)
            g_path = gtn.viterbi_path(gtn.intersect(g_emissions, g_transitions))
            prediction = g_path.labels_to_list()

            collapsed_prediction = [p for p, _ in groupby(prediction)]
            if self.garbage_idx is not None:
                # remove garbage tokens
                collapsed_prediction = [
                    p for p in collapsed_prediction if p != self.garbage_idx
                ]
            predictions[b] = unpack_replabels(collapsed_prediction, self.num_replabels)

        gtn.parallel_for(process, range(B))
        return [torch.IntTensor(p) for p in predictions]
