import logging
import numpy as np
import torch
import struct
import gtn
from concurrent.futures import ThreadPoolExecutor
import threading


def get_data_ptr_as_bytes(tensor, offset = 0):
    return struct.pack("P", tensor.data_ptr() + offset)

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
        num_workers=16)


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

class CTCLossFunction(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, log_probs, targets, blank_idx=0, reduction='none') : 
        grad_enabled = log_probs.requires_grad
        is_cuda = False
        if log_probs.is_cuda:
            is_cuda = True 
            log_probs = log_probs.cpu()
        B, T, C = list(log_probs.shape) 
        loss = torch.zeros(B, dtype=torch.float)
        input_grad = torch.zeros(log_probs.shape) if grad_enabled else torch.Tensor()
        def process(b):
            # create emission graph
            emissions = gtn.create_linear_graph(get_data_ptr_as_bytes(log_probs, b * T * C * 4), T, C, True)
               
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
            fwd_graph = gtn.forward(gtn.compose(emissions, criterion))
            scale = -1.0
            if reduction == "mean":
                scale = -1.0 / L if L > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid value for reduction '" +  str(reduction) + "'")   
            loss[b] = fwd_graph.item() * scale
            
            if grad_enabled:
                gtn.backward(fwd_graph)             
                gtn.extract_linear_grad(emissions, scale, get_data_ptr_as_bytes(input_grad, b * T * C * 4))
        
        # TODO: remove hard coding of max_workers
        executor = ThreadPoolExecutor(max_workers=10)
        for b in range(B):
            executor.submit(process, b)
        executor.shutdown()
        ctx.constant = input_grad.cuda() / B if is_cuda else input_grad / B
        return torch.mean(loss.cuda() if is_cuda else loss)  

    @staticmethod 
    def backward(ctx, grad_output) :
        return ctx.constant *  grad_output.view(-1,1,1).expand(ctx.constant.size()), None, None, None


CTCLoss=CTCLossFunction.apply 
