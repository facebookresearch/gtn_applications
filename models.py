import torch

class RNN(torch.nn.Module):

    def __init__(
            self, input_size, output_size, cell_type,
            hidden_size, num_layers, dropout, bidirectional):
        super(RNN, self).__init__()

        if cell_type.upper() not in ["RNN", "LSTM", "GRU"]:
            raise ValueError(f"Unkown rnn cell type {cell_type}")
        self.rnn = getattr(torch.nn, cell_type.upper())(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.linear = torch.nn.Linear(
            hidden_size + bidirectional * hidden_size, output_size)

    def forward(self, inputs):
        # inputs shape: [batch size, img height (e.g. input size), img width (e.g. sequence length)]
        # outputs shape: [img width, batch size, num classes]
        outputs = inputs.permute(2, 0, 1)
        outputs, _ = self.rnn(outputs)
        return self.linear(outputs)


def ctc(blank=0, pad_val=-1):
    ctc_crit = torch.nn.CTCLoss(blank=blank)
    def criterion(inputs, targets):
        input_lengths = torch.tensor([inputs.shape[0]] * inputs.shape[1])
        target_lengths = torch.sum(targets != pad_val, dim=1)
        log_probs = torch.nn.functional.log_softmax(inputs, dim=2)
        return ctc_crit(log_probs, targets, input_lengths, target_lengths)
    return criterion


def load_model(model_type, input_size, output_size, **kwargs):
    if model_type == "rnn":
        return RNN(
            input_size,
            output_size,
            kwargs["cell_type"],
            kwargs["hidden_size"],
            kwargs["num_layers"],
            kwargs.get("dropout", 0.0),
            kwargs.get("bidirectional", False))
    else:
        raise ValueError(f"Unknown model type {model_type}")

