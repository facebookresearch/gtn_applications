from concurrent.futures import ThreadPoolExecutor
import gtn
from itertools import groupby
import numpy as np
import torch
import utils


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
        self, input_size, output_size, depth, tds_groups, kernel_size, dropout
    ):
        super(TDS2d, self).__init__()
        # downsample layer -> TDS2d group -> ... -> Linear output layer
        modules = []
        in_channels = 1
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
        outputs = inputs.unsqueeze(1)
        outputs = self.tds(outputs)

        # outputs shape: [B, C, H, W]
        B, C, H, W = outputs.shape
        outputs = outputs.reshape(B, C * H, W)

        # outputs shape: [W, B, output_size]
        return self.linear(outputs.permute(2, 0, 1))


class TDSBlock(torch.nn.Module):
    def __init__(self, in_channels, img_height, kernel_size, dropout):
        super(TDSBlock, self).__init__()
        self.in_channels = in_channels
        self.img_height = img_height
        fc_size = in_channels * img_height
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
        C, H = self.in_channels, self.img_height
        outputs = self.conv(inputs.view(B, C, H, W)).view(B, CH, W) + inputs
        outputs = self.instance_norms[0](outputs)

        outputs = self.fc(outputs.transpose(1, 2)).transpose(1, 2) + outputs
        outputs = self.instance_norms[1](outputs)

        # outputs shape: [B, C * H, W]
        return outputs


class TDS(torch.nn.Module):
    def __init__(self, input_size, output_size, tds_groups, kernel_size, dropout):
        super(TDS, self).__init__()
        # TODO might be worth adding a 2D front-end or changing TDS to be a grouped conv
        # downsample layer -> TDS group -> ... -> Linear output layer
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
                        stride=2,
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
        # outputs shape: [W, B, output_size]
        return self.linear(outputs.permute(2, 0, 1))


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
        )
        self.linear = torch.nn.Linear(
            hidden_size + bidirectional * hidden_size, output_size
        )

    def forward(self, inputs):
        # inputs shape: [batch size, img height (e.g. input size), img width (e.g. sequence length)]
        # outputs shape: [img width, batch size, num classes]
        outputs = inputs.unsqueeze(1)
        outputs = self.convs(outputs)
        b, c, h, w = outputs.shape
        outputs = outputs.reshape(b, c * h, w)

        outputs = outputs.permute(2, 0, 1)
        outputs, _ = self.rnn(outputs)
        return self.linear(outputs)


class CTC(torch.nn.Module):
    def __init__(self, blank, use_pt):
        super(CTC, self).__init__()
        self.blank = blank  # index of blank label
        self.use_pt = use_pt  # use pytorch version instead of GTN

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=2)

        if not self.use_pt:
            targets = [t.tolist() for t in targets]
            log_probs = log_probs.permute(1, 0, 2)  # T x B X C ->  B x T x C
            return utils.CTCLoss(log_probs, targets, self.blank, "mean")
        else:
            input_lengths = [inputs.shape[0]] * inputs.shape[1]
            target_lengths = [t.numel() for t in targets]
            targets = torch.cat(targets)
            return torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=self.blank
            )

    def viterbi(self, outputs):
        predictions = torch.argmax(outputs, dim=2).T.to("cpu")
        collapsed_predictions = []
        for pred in predictions.split(1):
            pred = pred.squeeze()
            mask = pred[1:] != pred[:-1]
            pred = torch.cat([pred[0:1], pred[1:][mask]])
            pred = pred[pred != self.blank]
            collapsed_predictions.append(pred)
        return collapsed_predictions


class ASG(torch.nn.Module):
    def __init__(self, num_classes, num_replabels=1):
        super(ASG, self).__init__()
        self.num_classes = num_classes
        self.num_replabels = num_replabels
        assert self.num_replabels > 0
        self.transitions = torch.nn.Parameter(
            torch.zeros(
                self.num_classes + num_replabels + 1, self.num_classes + num_replabels
            )
        )

    def forward(self, inputs, targets):
        targets = [t.tolist() for t in targets]
        inputs = inputs.permute(1, 0, 2)  # T x B X C ->  B x T x C
        return utils.ASGLoss(inputs, self.transitions, targets, "mean")

    def viterbi(self, outputs):
        outputs = outputs.permute(1, 0, 2)  # T x B X C ->  B x T x C
        B, T, C = outputs.shape
        assert C == self.num_classes + self.num_replabels

        def process(b):
            prediction = []
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu(memory_format=torch.contiguous_format)
            g_emissions.set_weights(cpu_data.data_ptr())

            # create transition graph
            g_transitions = utils.ASGLossFunction.create_transitions_graph(
                self.transitions
            )
            g_path = gtn.viterbi_path(gtn.intersect(g_emissions, g_transitions))
            prediction = g_path.labels_to_list()
            return utils.unpack_replabels(prediction, self.num_replabels)

        collapsed_predictions = []
        executor = ThreadPoolExecutor(max_workers=B, initializer=utils.thread_init)
        futures = [executor.submit(process, b) for b in range(B)]
        for f in futures:
            prediction = torch.IntTensor(f.result())
            collapsed_predictions.append(prediction)
        executor.shutdown()

        return collapsed_predictions


def load_model(model_type, input_size, output_size, config):
    if model_type == "rnn":
        return RNN(input_size, output_size, **config)
    elif model_type == "tds":
        return TDS(input_size, output_size, **config)
    elif model_type == "tds2d":
        return TDS2d(input_size, output_size, **config)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def load_criterion(criterion_type, num_classes, config):
    if criterion_type == "asg":
        num_replabels = config.get("num_replabels", 0)
        return ASG(num_classes, num_replabels), num_classes + num_replabels
    elif criterion_type == "ctc":
        use_pt = config.get("use_pt", True)
        return CTC(num_classes, use_pt), num_classes + 1  # account for blank
    elif criterion_type == "transducer":
        criterion = transducer.Transducer(
            preprocessor.tokens,
            preprocessor.graphemes_to_index,
            blank=config["criterion"]["blank"],
            allow_repeats=config["criterion"]["allow_repeats"],
            reduction="mean")
    else:
        raise ValueError(f"Unknown model type {criterion_type}")
