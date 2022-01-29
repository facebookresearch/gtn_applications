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
from criterions import ctc, asg, transducer
from models import rnn, tds, tds2d


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
        lst = [
            self.loss,
            self.num_samples,
            self.num_tokens,
            self.edit_distance_tokens,
            self.num_words,
            self.edit_distance_words,
        ]
        # TODO: avoid this so that distributed cpu training also works
        lst_tensor = torch.FloatTensor(lst).cuda()
        torch.distributed.all_reduce(lst_tensor)
        (
            self.loss,
            self.num_samples,
            self.num_tokens,
            self.edit_distance_tokens,
            self.num_words,
            self.edit_distance_words,
        ) = lst_tensor.tolist()

    @property
    def avg_loss(self):
        return self.loss / self.num_samples if self.num_samples > 0 else 0

    @property
    def cer(self):
        return (
            self.edit_distance_tokens * 100.0 / self.num_tokens
            if self.num_tokens > 0
            else 0
        )

    @property
    def wer(self):
        return (
            self.edit_distance_words * 100.0 / self.num_words
            if self.num_words > 0
            else 0
        )


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


def load_model(model_type, input_size, output_size, config):
    if model_type == "rnn":
        return rnn.RNN(input_size, output_size, **config)
    elif model_type == "tds":
        return tds.TDS(input_size, output_size, **config)
    elif model_type == "tds2d":
        return tds2d.TDS2d(input_size, output_size, **config)
    elif model_type == "tds2d_transducer":
        return tds2d.TDS2dTransducer(input_size, output_size, **config)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def load_criterion(criterion_type, preprocessor, config):
    num_tokens = preprocessor.num_tokens
    if criterion_type == "asg":
        num_replabels = config.get("num_replabels", 0)
        use_garbage = config.get("use_garbage", True)
        return (
            asg.ASG(num_tokens, num_replabels, use_garbage),
            num_tokens + num_replabels + int(use_garbage),
        )
    elif criterion_type == "ctc":
        use_pt = config.get("use_pt", True)  # use pytorch implementation
        return ctc.CTC(num_tokens, use_pt), num_tokens + 1  # account for blank
    elif criterion_type == "transducer":
        blank = config.get("blank", "none")
        transitions = config.get("transitions", None)
        if transitions is not None:
            transitions = gtn.load(transitions)
        criterion = transducer.Transducer(
            preprocessor.tokens,
            preprocessor.graphemes_to_index,
            ngram=config.get("ngram", 0),
            transitions=transitions,
            blank=blank,
            allow_repeats=config.get("allow_repeats", True),
            reduction="mean",
        )
        return criterion, num_tokens + int(blank != "none")
    else:
        raise ValueError(f"Unknown model type {criterion_type}")


def load_from_checkpoint(model, criterion, checkpoint_path, load_last=False):
    model_checkpoint = os.path.join(checkpoint_path, "model.checkpoint")
    criterion_checkpoint = os.path.join(checkpoint_path, "criterion.checkpoint")
    if not load_last:
        model_checkpoint += ".best"
        criterion_checkpoint += ".best"
    model.load_state_dict(torch.load(model_checkpoint))
    criterion.load_state_dict(torch.load(criterion_checkpoint))
