import logging
import numpy as np
import torch


def data_loader(dataset, config):
    num_samples = config["data"].get("num_samples", None)
    if num_samples is not None:
        logging.info(f"Using {num_samples} of {len(dataset)}.")
        dataset = Subset(
            dataset, torch.randperm(len(dataset))[:num_samples])
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BatchSortedSampler(
            dataset, config["optim"]["batch_size"]),
            collate_fn=padding_collate,
        num_workers=32)


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

    def __init__(self, dataset, batch_size, shuffle=True):
        widths = (in_size[0] for in_size, _ in dataset.sample_sizes())
        sorted_dataset = sorted(enumerate(widths), key = lambda x: x[1])
        sorted_indices, _ = zip(*sorted_dataset)
        self.batches = [sorted_indices[idx:idx+batch_size]
            for idx in range(0, len(sorted_indices), batch_size)]
        self.shuffle = shuffle

    def __iter__(self):
        order = torch.randperm if self.shuffle else torch.arange
        return (self.batches[i] for i in order(len(self.batches)))

    def __len__(self):
        return len(self.batches)


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
