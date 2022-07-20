"""
Copyright (c) Meta Platforms, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn
import torch
import math

# blank idx is REQUIRED to be zero for current implementation
STC_BLANK_IDX = 0


class STCLossFunction(torch.autograd.Function):
    """
    Creates a function for STC with autograd
    NOTE: This function assumes <star>, <star>/token is appended to the input
    """

    @staticmethod
    def create_stc_graph(target, star_idx, prob):
        """
        Creates STC label graph

        Attributes:
            target: initial value for token insertion penalty (before applying log)
            star_idx: index of star token
            prob: token insertion penalty (before applying log)
        Returns:
            STC label graph as gtn.Graph
        """
        g = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        # create self-less CTC graph
        for l in range(S):
            idx = (l - 1) // 2
            g.add_node(l == 0, l == S - 1 or l == S - 2)
            label = target[idx] if l % 2 else STC_BLANK_IDX
            if label == STC_BLANK_IDX:
                g.add_arc(l, l, label)
            if l > 0:
                g.add_arc(l - 1, l, label)
            if l % 2 and l > 1:
                g.add_arc(l - 2, l, label)

        # add extra nodes/arcs required for STC
        for l in range(L + 1):
            p1 = 2 * l - 1
            p2 = 2 * l

            c1 = g.add_node(False, l == L)
            idx = star_idx if l == L else (star_idx + target[l])
            if p1 >= 0:
                g.add_arc(p1, c1, idx, idx, math.log(prob))
            g.add_arc(p2, c1, idx, idx, math.log(prob))
            g.add_arc(c1, c1, idx, idx, math.log(prob))
            if l < L:
                g.add_arc(c1, 2 * l + 1, target[l])
            g.add_arc(c1, p2, STC_BLANK_IDX)

        return g

    @staticmethod
    def forward(ctx, inputs, targets, prob, reduction="none"):
        B, T, Cstar = inputs.shape
        losses, scales, emissions_graphs = [None] * B, [None] * B, [None] * B
        C = Cstar // 2

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(
                T, Cstar, gtn.Device(gtn.CPU), inputs.requires_grad
            )
            cpu_data = inputs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create criterion graph
            g_criterion = STCLossFunction.create_stc_graph(targets[b], C, prob)
            g_criterion.arc_sort(False)

            # compose the graphs
            g_loss = gtn.negate(
                gtn.forward_score(gtn.compose(g_criterion, g_emissions))
            )

            scale = 1.0
            if reduction == "mean":
                scale = 1.0 / T if T > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid value for reduction '" + str(reduction) + "'")

            # Save for backward:
            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions

        gtn.parallel_for(process, range(B))

        ctx.auxiliary_data = (losses, scales, emissions_graphs, inputs.shape)
        loss = torch.tensor([losses[b].item() * scales[b] for b in range(B)])
        return torch.mean(loss.cuda() if inputs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        losses, scales, emissions_graphs, in_shape = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C))

        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            input_grad[b] = torch.from_numpy(grad).view(1, T, C) * scales[b]

        gtn.parallel_for(process, range(B))

        if grad_output.is_cuda:
            input_grad = input_grad.cuda()
        input_grad *= grad_output / B

        return (
            input_grad,
            None,  # targets
            None,  # prob
            None,  # reduction
        )


STCLoss = STCLossFunction.apply


class STC(torch.nn.Module):
    """The Star Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a
    partially labeled target sequence.

    Attributes:
        p0: initial value for token insertion penalty (before applying log)
        plast: final value for token insertion penalty (before applying log)
        thalf: number of steps for token insertion penalty (before applying log)
            to reach (p0 + plast)/2
    """

    def __init__(self, blank_idx, p0=1, plast=1, thalf=1, reduction="none"):
        super(STC, self).__init__()
        assert blank_idx == STC_BLANK_IDX
        self.p0 = p0
        self.plast = plast
        self.thalf = thalf
        self.nstep = 0
        self.reduction = reduction

    @staticmethod
    def logsubexp(a, b):
        """
        Computes log(exp(a) - exp(b))

        Args:
            a: Tensor of size (M x N)
            b: Tensor of size (M x N x O)
        Returns:
            Tensor of size (M x N x O)
        """

        with torch.set_grad_enabled(a.requires_grad):
            B, T, C = b.shape
            a = a.tile((1, 1, C))
            return a + torch.log1p(1e-7 - torch.exp(b - a))

    def forward(self, inputs, targets):
        """
        Computes STC loss for the given input and partialy labeled target

        Args:
            inputs: Tensor of size (T, B, C)
                T - # time steps, B - batch size, C - alphabet size (including blank)
                The logarithmized probabilities of the outputs (e.g. obtained with torch.nn.functional.log_softmax())
            targets: list of size [B]
                List of target sequences for each batch

        Returns:
            Tensor of size 1
            Mean STC loss of all samples in the batch
        """

        if self.training:
            self.nstep += 1

        prob = self.plast + (self.p0 - self.plast) * math.exp(
            -self.nstep * math.log(2) / self.thalf
        )
        # (T, B, C) --> (B, T, C)
        log_probs = inputs.permute(1, 0, 2)

        B, T, C = log_probs.shape
        with torch.set_grad_enabled(log_probs.requires_grad):
            # <star>
            lse = torch.logsumexp(log_probs[:, :, 1:], 2, keepdim=True)

            # select only the tokens present in current batch
            select_idx = [STC_BLANK_IDX] + list(
                set([t for target in targets for t in target])
            )
            target_map = {}
            for i, t in enumerate(select_idx):
                target_map[t] = i

            select_idx = torch.IntTensor(select_idx).to(log_probs.device)
            log_probs = log_probs.index_select(2, select_idx)
            targets = [[target_map[t] for t in target] for target in targets]

            # <star>\tokens for all tokens present in current batch
            neglse = STC.logsubexp(lse, log_probs[:, :, 1:])

            # concatenate (tokens, <star>, <star>\tokens)
            log_probs = torch.cat([log_probs, lse, neglse], dim=2)
        return STCLoss(log_probs, targets, prob, self.reduction)
