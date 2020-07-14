from concurrent.futures import ThreadPoolExecutor
import collections
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


def data_loader(dataset, config, world_rank, world_size):
    num_samples = config["data"].get("num_samples", None)
    if num_samples is not None:
        logging.info(f"Using {num_samples} of {len(dataset)}.")
        dataset = Subset(dataset, torch.randperm(len(dataset))[:num_samples])
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BatchSortedSampler(dataset,
                                         config["optim"]["batch_size"],
                                         world_rank, world_size),
        collate_fn=padding_collate,
        num_workers=1,
    )


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
    def __init__(self,
                 dataset,
                 batch_size,
                 world_rank,
                 world_size,
                 shuffle=True):
        local_batchsize = batch_size // world_size
        widths = (in_size[0] for in_size, _ in dataset.sample_sizes())
        sorted_dataset = sorted(enumerate(widths), key=lambda x: x[1])
        sorted_indices, _ = zip(*sorted_dataset)
        global_batches = [
            sorted_indices[idx:idx + local_batchsize]
            for idx in range(0, len(sorted_indices), local_batchsize)
        ]
        self.length = len(global_batches) // world_size
        # distribute the sample across the ranks
        self.batches = [
            global_batches[world_rank + i * world_size]
            for i in range(self.length)
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
    batch_inputs = torch.zeros(
        (len(inputs), inputs[0].shape[1], max_input_len))
    for e, ip in enumerate(inputs):
        batch_inputs[e, :, :ip.shape[2]] = ip

    return batch_inputs, targets


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
        return {k : self.running_times[k] / self.n[k] for k in self.keys}

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
            self.running_times[k] += (time * 1e-3)
            self.n[k] += len(starts)
        self.start_events = collections.defaultdict(list)
        self.end_events = collections.defaultdict(list)


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
    def forward(ctx, log_probs, targets, blank_idx=0, reduction="none"):
        grad_enabled = log_probs.requires_grad
        is_cuda = False
        if log_probs.is_cuda:
            is_cuda = True
            log_probs = log_probs.cpu()
        B, T, C = list(log_probs.shape)
        loss = torch.zeros(B, dtype=torch.float)
        input_grad = torch.zeros(
            log_probs.shape) if grad_enabled else torch.Tensor()

        def process(b):
            # create emission graph
            emissions = gtn.linear_graph(T, C, True)
            emissions.set_weights(log_probs[b].flatten().tolist())

            # create criterion graph
            criterion = gtn.Graph(False)
            target = targets[b]
            L = len(target)
            S = 2 * L + 1
            for l in range(S):
                idx = (l - 1) // 2
                criterion.add_node(l == 0, l == S - 1 or l == S - 2)
                label = target[idx] if l % 2 else blank_idx
                criterion.add_arc(l, l, label)
                if l > 0:
                    criterion.add_arc(l - 1, l, label)
                if l % 2 and l > 1 and label != target[idx - 1]:
                    criterion.add_arc(l - 2, l, label)

            # compose the graphs
            fwd_graph = gtn.forward_score(gtn.compose(emissions, criterion))
            scale = -1.0
            if reduction == "mean":
                scale = -1.0 / L if L > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid value for reduction '" +
                                 str(reduction) + "'")
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


CTCLoss = CTCLossFunction.apply
