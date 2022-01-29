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
from criterions import transducer


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
        self,
        input_size,
        output_size,
        depth,
        tds_groups,
        kernel_size,
        dropout,
        in_channels=1,
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
        self,
        input_size,
        output_size,
        tokens,
        kernel_size,
        stride,
        tds1,
        tds2,
        wfst=True,
        **kwargs,
    ):
        super(TDS2dTransducer, self).__init__()
        # TDS2d -> ConvTransducer -> TDS2d

        # Setup lexicon for transducer layer:
        with open(tokens, "r") as fid:
            output_tokens = [l.strip() for l in fid]
        input_tokens = set(t for token in output_tokens for t in token)
        input_tokens = {t: e for e, t in enumerate(sorted(input_tokens))}
        lexicon = [tuple(input_tokens[t] for t in token) for token in output_tokens]
        in_token_size = len(input_tokens) + 1
        blank_idx = len(input_tokens)

        # output size of tds1 is number of input tokens + 1 for blank
        self.tds1 = TDS2d(input_size, in_token_size, **tds1)
        stride_h = np.prod([grp["stride"][0] for grp in tds1["tds_groups"]])
        inner_size = input_size // stride_h

        # output size of conv is the size of the lexicon
        if wfst:
            self.conv = transducer.ConvTransduce1D(
                lexicon, kernel_size, stride, blank_idx, **kwargs
            )
        else:
            # For control, use "dumb" conv with the same parameters as the WFST conv:
            self.conv = torch.nn.Conv1d(
                in_channels=in_token_size,
                out_channels=len(lexicon),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            )
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
