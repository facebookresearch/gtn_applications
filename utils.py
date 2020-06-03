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

    return batch_inputs, targets


def edit_distance(ref, hyp):
    """
    Edit distance between two sequences reference (ref) and hypothesis (hyp).
    Returns edit distance, number of insertions, deletions and substitutions to
    transform hyp to ref, and number of correct matches.
    """
    n = len(ref)
    m = len(hyp)

    ins = dels = subs = corr = 0

    D = torch.zeros((n+1, m+1), dtype=torch.long)

    D[:,0] = torch.arange(n+1)
    D[0,:] = torch.arange(m+1)

    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j], D[i,j-1], D[i-1,j-1]) + 1

    i = n
    j = m
    while i > 0 and j > 0:
        if ref[i-1] == hyp[j-1]:
            corr += 1
        elif D[i-1,j] == D[i,j]-1:
            ins += 1
            j += 1
        elif D[i,j-1] == D[i,j]-1:
            dels += 1
            i += 1
        elif D[i-1,j-1] == D[i,j]-1:
            subs += 1
        i -= 1
        j -= 1

    ins += i
    dels += j

    return D[-1,-1], (ins, dels, subs, corr)
