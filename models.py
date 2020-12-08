"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn
from itertools import groupby
import numpy as np
import os
import torch
import utils
import transducer


class TDSBlock2d(torch.nn.Module):
    def __init__(self, in_channels, img_depth, kernel_size, dropout):
        super(TDSBlock2d, self).__init__()
        self.in_channels = in_channels
        self.img_depth = img_depth
        fc_size = in_channels * img_depth
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, kernel_size[0], kernel_size[1]),
                padding=(0, kernel_size[0] // 2, kernel_size[1] // 2),
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.Dropout(dropout),
        )
        self.instance_norms = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm2d(fc_size, affine=True),
                torch.nn.InstanceNorm2d(fc_size, affine=True),
            ]
        )

    def forward(self, inputs):
        # inputs shape: [B, CD, H, W]
        B, CD, H, W = inputs.shape
        C, D = self.in_channels, self.img_depth
        outputs = self.conv(inputs.view(B, C, D, H, W)).view(B, CD, H, W) + inputs
        outputs = self.instance_norms[0](outputs)

        outputs = self.fc(outputs.transpose(1, 3)).transpose(1, 3) + outputs
        outputs = self.instance_norms[1](outputs)

        # outputs shape: [B, CD, H, W]
        return outputs


class TDS2d(torch.nn.Module):
    def __init__(
        self, input_size, output_size, depth, tds_groups, kernel_size, dropout,
        in_channels=1
    ):
        super(TDS2d, self).__init__()
        # downsample layer -> TDS2d group -> ... -> Linear output layer
        self.in_channels = in_channels
        modules = []
        stride_h = np.prod([grp["stride"][0] for grp in tds_groups])
        assert (
            input_size % stride_h == 0
        ), f"Image height not divisible by total stride {stride_h}."
        for tds_group in tds_groups:
            # add downsample layer:
            out_channels = depth * tds_group["channels"]
            modules.extend(
                [
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                        stride=tds_group["stride"],
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.InstanceNorm2d(out_channels, affine=True),
                ]
            )
            for _ in range(tds_group["num_blocks"]):
                modules.append(
                    TDSBlock2d(tds_group["channels"], depth, kernel_size, dropout)
                )
            in_channels = out_channels
        self.tds = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(in_channels * input_size // stride_h, output_size)

    def forward(self, inputs):
        # inputs shape: [B, H, W]
        B, H, W = inputs.shape
        outputs = inputs.reshape(B, self.in_channels, H // self.in_channels, W)
        outputs = self.tds(outputs)

        # outputs shape: [B, C, H, W]
        B, C, H, W = outputs.shape
        outputs = outputs.reshape(B, C * H, W)

        # outputs shape: [B, W, output_size]
        return self.linear(outputs.permute(0, 2, 1))


class TDS2dTransducer(torch.nn.Module):
    def __init__(
            self, input_size, output_size, tokens, kernel_size, stride, tds1, tds2, wfst=True, **kwargs,
    ):
        super(TDS2dTransducer, self).__init__()
        # TDS2d -> ConvTransducer -> TDS2d

        # Setup lexicon for transducer layer:
        with open(tokens, 'r') as fid:
            output_tokens = [l.strip() for l in fid]
        input_tokens = set(t for token in output_tokens for t in token)
        input_tokens = {t: e for e, t in enumerate(sorted(input_tokens))}
        lexicon = [tuple(input_tokens[t] for t in token)
                    for token in output_tokens]
        in_token_size = len(input_tokens) + 1
        blank_idx = len(input_tokens)

        # output size of tds1 is number of input tokens + 1 for blank
        self.tds1 = TDS2d(input_size, in_token_size, **tds1)
        stride_h = np.prod([grp["stride"][0] for grp in tds1["tds_groups"]])
        inner_size = input_size // stride_h

        # output size of conv is the size of the lexicon
        if wfst:
            self.conv = transducer.ConvTransduce1D(
                lexicon, kernel_size, stride, blank_idx, **kwargs)
        else:
            # For control, use "dumb" conv with the same parameters as the WFST conv:
            self.conv = torch.nn.Conv1d(
                in_channels=in_token_size,
                out_channels=len(lexicon),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride)
        self.wfst = wfst

        # in_channels should be set to out_channels of prevous tds group * depth
        in_channels = tds1["tds_groups"][-1]["channels"] * tds1["depth"]
        tds2["in_channels"] = in_channels
        self.linear = torch.nn.Linear(len(lexicon), in_channels * inner_size)
        self.tds2 = TDS2d(inner_size, output_size, **tds2)

    def forward(self, inputs):
        # inputs shape: [B, H, W]
        outputs = self.tds1(inputs)
        # outputs shape: [B, W, C]
        if self.wfst:
            outputs = self.conv(outputs)
        else:
            outputs = self.conv(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        # outputs shape: [B, W, C']
        outputs = self.linear(outputs)
        return self.tds2(outputs.permute(0, 2, 1))


class TDSBlock(torch.nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, dropout):
        super(TDSBlock, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        fc_size = in_channels * num_features
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.Dropout(dropout),
        )
        self.instance_norms = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(fc_size, affine=True),
                torch.nn.InstanceNorm1d(fc_size, affine=True),
            ]
        )

    def forward(self, inputs):
        # inputs shape: [B, C * H, W]
        B, CH, W = inputs.shape
        C, H = self.in_channels, self.num_features
        outputs = self.conv(inputs.view(B, C, H, W)).view(B, CH, W) + inputs
        outputs = self.instance_norms[0](outputs)

        outputs = self.fc(outputs.transpose(1, 2)).transpose(1, 2) + outputs
        outputs = self.instance_norms[1](outputs)

        # outputs shape: [B, C * H, W]
        return outputs


class TDS(torch.nn.Module):
    def __init__(self, input_size, output_size, tds_groups, kernel_size, dropout):
        super(TDS, self).__init__()
        modules = []
        in_channels = input_size
        for tds_group in tds_groups:
            # add downsample layer:
            out_channels = input_size * tds_group["channels"]
            modules.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=tds_group.get("stride", 2),
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.InstanceNorm1d(out_channels, affine=True),
                ]
            )
            for _ in range(tds_group["num_blocks"]):
                modules.append(
                    TDSBlock(tds_group["channels"], input_size, kernel_size, dropout)
                )
            in_channels = out_channels
        self.tds = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(in_channels, output_size)

    def forward(self, inputs):
        # inputs shape: [B, H, W]
        outputs = self.tds(inputs)
        # outputs shape: [B, W, output_size]
        return self.linear(outputs.permute(0, 2, 1))


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        cell_type,
        hidden_size,
        num_layers,
        dropout=0.0,
        bidirectional=False,
        channels=[8, 8],
        kernel_sizes=[[5, 5], [5, 5]],
        strides=[[2, 2], [2, 2]],
    ):
        super(RNN, self).__init__()

        # convolutional front-end:
        convs = []
        in_channels = 1
        for out_channels, kernel, stride in zip(channels, kernel_sizes, strides):
            padding = (kernel[0] // 2, kernel[1] // 2)
            convs.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            convs.append(torch.nn.ReLU())
            if dropout > 0:
                convs.append(torch.nn.Dropout(dropout))
            in_channels = out_channels

        self.convs = torch.nn.Sequential(*convs)
        rnn_input_size = input_size * out_channels

        if cell_type.upper() not in ["RNN", "LSTM", "GRU"]:
            raise ValueError(f"Unkown rnn cell type {cell_type}")
        self.rnn = getattr(torch.nn, cell_type.upper())(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(
            hidden_size + bidirectional * hidden_size, output_size
        )

    def forward(self, inputs):
        # inputs shape: [B, H, W]
        outputs = inputs.unsqueeze(1)
        outputs = self.convs(outputs)
        b, c, h, w = outputs.shape
        outputs = outputs.reshape(b, c * h, w).permute(0, 2, 1)
        outputs, _ = self.rnn(outputs)
        # outputs shape: [B, W, output_size]
        return self.linear(outputs)


class CTC(torch.nn.Module):
    def __init__(self, blank, use_pt):
        super(CTC, self).__init__()
        self.blank = blank  # index of blank label
        self.use_pt = use_pt  # use pytorch version instead of GTN

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=2)

        if self.use_pt:
            log_probs = log_probs.permute(1, 0, 2)  # expects [T, B, C]
            input_lengths = [inputs.shape[1]] * inputs.shape[0]
            target_lengths = [t.numel() for t in targets]
            targets = torch.cat(targets)
            return torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=self.blank, zero_infinity=True,
            )
        else:
            targets = [t.tolist() for t in targets]
            return utils.CTCLoss(log_probs, targets, self.blank, "mean")

    def viterbi(self, outputs):
        predictions = torch.argmax(outputs, dim=2).to("cpu")
        collapsed_predictions = []
        for pred in predictions.split(1):
            pred = pred.squeeze(0)
            mask = pred[1:] != pred[:-1]
            pred = torch.cat([pred[0:1], pred[1:][mask]])
            pred = pred[pred != self.blank]
            collapsed_predictions.append(pred)
        return collapsed_predictions


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
        targets = [
            utils.pack_replabels(t.tolist(), self.num_replabels) for t in targets
        ]
        if self.garbage_idx is not None:
            # add a garbage token between each target label
            for idx in range(len(targets)):
                prev_tgt = targets[idx]
                targets[idx] = [self.garbage_idx] * (len(prev_tgt) * 2 + 1)
                targets[idx][1::2] = prev_tgt
        return utils.ASGLoss(inputs, self.transitions, targets, "mean")

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
            g_transitions = utils.ASGLossFunction.create_transitions_graph(
                self.transitions
            )
            g_path = gtn.viterbi_path(gtn.intersect(g_emissions, g_transitions))
            prediction = g_path.labels_to_list()

            collapsed_prediction = [p for p, _ in groupby(prediction)]
            if self.garbage_idx is not None:
                # remove garbage tokens
                collapsed_prediction = [
                    p for p in collapsed_prediction if p != self.garbage_idx
                ]
            predictions[b] = utils.unpack_replabels(
                collapsed_prediction, self.num_replabels)

        gtn.parallel_for(process, range(B))
        return [torch.IntTensor(p) for p in predictions]


def load_model(model_type, input_size, output_size, config):
    if model_type == "rnn":
        return RNN(input_size, output_size, **config)
    elif model_type == "tds":
        return TDS(input_size, output_size, **config)
    elif model_type == "tds2d":
        return TDS2d(input_size, output_size, **config)
    elif model_type == "tds2d_transducer":
        return TDS2dTransducer(input_size, output_size, **config)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def load_criterion(criterion_type, preprocessor, config):
    num_tokens = preprocessor.num_tokens
    if criterion_type == "asg":
        num_replabels = config.get("num_replabels", 0)
        use_garbage = config.get("use_garbage", True)
        return (
            ASG(num_tokens, num_replabels, use_garbage),
            num_tokens + num_replabels + int(use_garbage),
        )
    elif criterion_type == "ctc":
        use_pt = config.get("use_pt", True) # use pytorch implementation
        return CTC(num_tokens, use_pt), num_tokens + 1  # account for blank
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
