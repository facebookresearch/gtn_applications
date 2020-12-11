"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import collections
from dataclasses import dataclass
import gtn
import importlib
import logging
import numpy as np
import os
import struct
import sys
import time
import torch

def data_loader(dataset, config, world_rank=0, world_size=1):
    num_samples = config["data"].get("num_samples", None)
    if num_samples is not None:
        logging.info(f"Using {num_samples} of {len(dataset)}.")
        dataset = Subset(dataset, torch.randperm(len(dataset))[:num_samples])
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BatchSortedSampler(
            dataset, config["optim"]["batch_size"], world_rank, world_size
        ),
        collate_fn=padding_collate,
        num_workers=int(world_size > 1),
    )

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

class Subset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super(Subset, self).__init__(dataset, indices)

    def sample_sizes(self):
        """
        Returns a list of tuples containing the input size
        (width, height) and the output length for each sample.
        """
        sizes = list(self.dataset.sample_sizes())
        for idx in self.indices:
            yield sizes[idx]


class BatchSortedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, world_rank, world_size, shuffle=True):
        local_batchsize = batch_size // world_size
        widths = (in_size[0] for in_size, _ in dataset.sample_sizes())
        sorted_dataset = sorted(enumerate(widths), key=lambda x: x[1])
        sorted_indices, _ = zip(*sorted_dataset)
        global_batches = [
            sorted_indices[idx : idx + local_batchsize]
            for idx in range(0, len(sorted_indices), local_batchsize)
        ]
        self.length = len(global_batches) // world_size
        # distribute the sample across the ranks
        self.batches = [
            global_batches[world_rank + i * world_size] for i in range(self.length)
        ]
        self.shuffle = shuffle

    def __iter__(self):
        order = torch.randperm if self.shuffle else torch.arange
        return (self.batches[i] for i in order(self.length))

    def __len__(self):
        return self.length


def padding_collate(samples):
    inputs, targets = zip(*samples)

    # collate inputs:
    h = inputs[0].shape[1]
    max_input_len = max(ip.shape[2] for ip in inputs)
    batch_inputs = torch.zeros((len(inputs), inputs[0].shape[1], max_input_len))
    for e, ip in enumerate(inputs):
        batch_inputs[e, :, : ip.shape[2]] = ip

    return batch_inputs, targets


@dataclass
class Meters:
    loss = 0.0
    num_samples = 0
    num_tokens = 0
    edit_distance_tokens = 0
    num_words = 0
    edit_distance_words = 0

    def sync(self):
        lst = [self.loss, self.num_samples, self.num_tokens, self.edit_distance_tokens, self.num_words, self.edit_distance_words]
        # TODO: avoid this so that distributed cpu training also works
        lst_tensor = torch.FloatTensor(lst).cuda()
        torch.distributed.all_reduce(lst_tensor)
        (
            self.loss,
            self.num_samples,
            self.num_tokens,
            self.edit_distance_tokens,
            self.num_words,
            self.edit_distance_words
        ) = lst_tensor.tolist()

    @property
    def avg_loss(self):
        return self.loss / self.num_samples if self.num_samples > 0 else 0

    @property
    def cer(self):
        return self.edit_distance_tokens * 100.0 / self.num_tokens if self.num_tokens > 0 else 0
    
    @property
    def wer(self):
        return self.edit_distance_words * 100.0 / self.num_words if self.num_words > 0 else 0


# A simple timer class inspired from `tnt.TimeMeter`
class CudaTimer:
    def __init__(self, keys):
        self.keys = keys
        self.reset()

    def start(self, key):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        self.start_events[key].append(s)
        return self

    def stop(self, key):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.end_events[key].append(e)
        return self

    def reset(self):
        self.start_events = collections.defaultdict(list)
        self.end_events = collections.defaultdict(list)
        self.running_times = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        return self

    def value(self):
        self._synchronize()
        return {k: self.running_times[k] / self.n[k] for k in self.keys}

    def _synchronize(self):
        torch.cuda.synchronize()
        for k in self.keys:
            starts = self.start_events[k]
            ends = self.end_events[k]
            if len(starts) == 0:
                raise ValueError("Trying to divide by zero in TimeMeter")
            if len(ends) != len(starts):
                raise ValueError("Call stop before checking value!")
            time = 0
            for start, end in zip(starts, ends):
                time += start.elapsed_time(end)
            self.running_times[k] += time * 1e-3
            self.n[k] += len(starts)
        self.start_events = collections.defaultdict(list)
        self.end_events = collections.defaultdict(list)


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


# Used to measure the time taken for multiple events
class Timer:
    def __init__(self, keys):
        self.keys = keys
        self.n = {}
        self.running_time = {}
        self.total_time = {}
        self.reset()

    def start(self, key):
        self.running_time[key] = time.time()
        return self

    def stop(self, key):
        self.total_time[key] = time.time() - self.running_time[key]
        self.n[key] += 1
        self.running_time[key] = None
        return self

    def reset(self):
        for k in self.keys:
            self.total_time[k] = 0
            self.running_time[k] = None
            self.n[k] = 0
        return self

    def value(self):
        vals = {}
        for k in self.keys:
            if self.n[k] == 0:
                raise ValueError("Trying to divide by zero in TimeMeter")
            else:
                vals[k] = self.total_time[k] / self.n[k]
        return vals


class CTCLossFunction(torch.autograd.Function):
    @staticmethod
    def create_ctc_graph(target, blank_idx):
        g_criterion = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        for l in range(S):
            idx = (l - 1) // 2
            g_criterion.add_node(l == 0, l == S - 1 or l == S - 2)
            label = target[idx] if l % 2 else blank_idx
            g_criterion.add_arc(l, l, label)
            if l > 0:
                g_criterion.add_arc(l - 1, l, label)
            if l % 2 and l > 1 and label != target[idx - 1]:
                g_criterion.add_arc(l - 2, l, label)
        g_criterion.arc_sort(False)
        return g_criterion

    @staticmethod
    def forward(ctx, log_probs, targets, blank_idx=0, reduction="none"):
        B, T, C = log_probs.shape
        losses = [None] * B
        scales = [None] * B
        emissions_graphs = [None] * B

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create criterion graph
            g_criterion = CTCLossFunction.create_ctc_graph(targets[b], blank_idx)
            # compose the graphs
            g_loss = gtn.negate(
                gtn.forward_score(gtn.intersect(g_emissions, g_criterion))
            )

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
            None,  # blank_idx
            None,  # reduction
        )


CTCLoss = CTCLossFunction.apply


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
