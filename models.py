import torch

class RNN(torch.nn.Module):

    def __init__(
            self, input_size, output_size, cell_type,
            hidden_size, num_layers,
            dropout=0.0, bidirectional=False,
            channels=[8, 8],
            kernel_sizes=[[5, 5], [5, 5]],
            strides=[[2, 2], [2, 2]]
            ):
        super(RNN, self).__init__()

        # convolutional front-end:
        convs = []
        in_channels = 1
        for out_channels, kernel, stride in zip(
                channels, kernel_sizes, strides):
            padding = (kernel[0] // 2, kernel[1] // 2)
            convs.append(torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding))
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
            bidirectional=bidirectional)
        self.linear = torch.nn.Linear(
            hidden_size + bidirectional * hidden_size, output_size)

    def forward(self, inputs):
        # inputs shape: [batch size, img height (e.g. input size), img width (e.g. sequence length)]
        # outputs shape: [img width, batch size, num classes]
        outputs = inputs.unsqueeze(1)
        outputs = self.convs(outputs)
        b, c, h, w = outputs.shape
        outputs = outputs.reshape(b, c*h, w)

        outputs = outputs.permute(2, 0, 1)
        outputs, _ = self.rnn(outputs)
        return self.linear(outputs)


class CTC(torch.nn.Module):
    def __init__(self, blank=0):
        super(CTC, self).__init__()
        self.blank = blank

    def forward(self, inputs, targets):
        input_lengths = [inputs.shape[0]] * inputs.shape[1]
        target_lengths = [t.numel() for t in targets]
        log_probs = torch.nn.functional.log_softmax(inputs, dim=2)
        targets = torch.cat(targets)
        return torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=self.blank)

    def decode(self, outputs):
        predictions = torch.argmax(outputs, dim=2).T.to("cpu")
        collapsed_predictions = []
        for pred in predictions.split(1):
            pred = pred.squeeze()
            mask = pred[1:] != pred[:-1]
            pred = torch.cat([pred[0:1], pred[1:][mask]])
            pred = pred[pred != self.blank]
            collapsed_predictions.append(pred)
        return collapsed_predictions


def load_model(model_type, input_size, output_size, config):
    if model_type == "rnn":
        return RNN(input_size, output_size, **config)
    else:
        raise ValueError(f"Unknown model type {model_type}")

