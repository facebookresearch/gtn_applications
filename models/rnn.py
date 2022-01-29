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
        h_out = input_size
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
            h_out //= stride
        self.convs = torch.nn.Sequential(*convs)
        rnn_input_size = h_out * out_channels

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
