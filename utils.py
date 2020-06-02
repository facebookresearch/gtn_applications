import torch


def data_loader(dataset, config):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BatchSortedSampler(
            dataset, config["optim"]["batch_size"]),
            collate_fn=padding_collate,
        num_workers=1)


class BatchSortedSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        widths = (in_size[1] for in_size, _ in dataset.sample_sizes())
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

    # collate targets:
    max_target_len = max(t.shape[0] for t in targets)
    batch_targets = torch.full(
        (len(targets), max_target_len), -1, dtype=targets[0].dtype)
    for e, t in enumerate(targets):
        batch_targets[e, :t.shape[0]] = t
    return batch_inputs, batch_targets
