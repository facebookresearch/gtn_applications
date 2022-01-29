"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn
import torch
import math


class STCLossFunction(torch.autograd.Function):
    @staticmethod
    def create_stc_graph(target, blank_idx, star_idx, wt):
        g = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        for l in range(S):
            idx = (l - 1) // 2
            g.add_node(l == 0, l == S - 1 or l == S - 2)
            label = target[idx] if l % 2 else blank_idx
            if label == blank_idx:
                g.add_arc(l, l, label)
            if l > 0:
                g.add_arc(l - 1, l, label)
            if l % 2 and l > 1 and label != target[idx - 1]:
                g.add_arc(l - 2, l, label)
        for l in range(L + 1):
            p1 = 2 * l - 1
            p2 = 2 * l

            c1 = g.add_node(False, l == L)
            idx = star_idx if l == L else (star_idx + target[l])
            if p1 >= 0:
                g.add_arc(p1, c1, idx, idx, math.log(wt))
            g.add_arc(p2, c1, idx, idx, math.log(wt))
            g.add_arc(c1, c1, idx, idx, math.log(wt))
            if l < L:
                g.add_arc(c1, 2 * l + 1, target[l])
            g.add_arc(c1, p2, blank_idx)
        g.arc_sort(False)
        return g

    @staticmethod
    def forward(ctx, log_probs, targets, wt, blank_idx=0, reduction="none"):
        B, T, C = log_probs.shape
        losses = [None] * B
        scales = [None] * B
        emissions_graphs = [None] * B

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(
                T, C, gtn.Device(gtn.CPU), log_probs.requires_grad
            )
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create criterion graph
            g_criterion = STCLossFunction.create_stc_graph(targets[b], blank_idx, C, wt)
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

        ctx.auxiliary_data = (losses, scales, emissions_graphs, log_probs.shape)
        loss = torch.tensor([losses[b].item() * scales[b] for b in range(B)])
        return torch.mean(loss.cuda() if log_probs.is_cuda else loss)

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
            None,  # wt
            None,  # blank_idx
            None,  # reduction
        )


STCLoss = STCLossFunction.apply


class STC(torch.nn.Module):
    def __init__(self, blank, w0, wlast, thalf):
        super(STC, self).__init__()
        assert blank == 0
        self.blank = blank  # index of blank label
        self.w0 = w0
        self.wlast = wlast
        self.thalf = thalf
        self.nstep = 0

    @staticmethod
    def logsubexp(a, b):
        with torch.set_grad_enabled(True):
            B, T, C = b.shape
            a = a.tile((1, 1, C))
            return a + torch.log1p(1e-7 - torch.exp(b - a))

    def forward(self, inputs, targets):
        if self.training:
            self.nstep += 1

        wt = self.wlast + (self.w0 - self.wlast) * math.exp(
            -self.nstep * math.log(2) / self.thalf
        )
        log_probs = inputs.permute(1, 0, 2)

        B, T, C = log_probs.shape
        with torch.set_grad_enabled(log_probs.requires_grad):
            lse = torch.logsumexp(log_probs[:, :, 1:], 2, keepdim=True)
            neglse = STC.logsubexp(lse, log_probs[:, :, 1:])
            log_probs = torch.cat([log_probs, lse, neglse], dim=2)
        return STCLoss(log_probs, targets, wt, self.blank, "mean")
