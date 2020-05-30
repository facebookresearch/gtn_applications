import torch

import datasets.iamdb


def test():
    pass


def train(model, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss = loss.item()
        test(model, val_loader)


def main():
    batch_size = 32

    # Setup data loader:
    preprocessor = datasets.iamdb.Preprocessor("datasets/data")
    trainset = datasets.iamdb.IamDB("datasets/data", preprocessor, split="train")
    valset = datasets.iamdb.IamDB("datasets/data", preprocessor, split="validation")
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1)

    # Setup Model:
    model = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional)

    # Run training:
    train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
