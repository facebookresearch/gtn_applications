import argparse
import json
import time
import torch

import datasets
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--config', type=str,
        help='A json configuration file for experiment.')
    parser.add_argument('--disable_cuda', action='store_true',
        help='Disable CUDA')
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    return args


def test():
    pass


def train(
        model, criterion, train_loader, valid_loader,
        epochs, lr, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            iter_time = time.time() - start_time
            print(inputs.shape)
            print(
                "Batch {}/{}: Loss {:.3f}, CER {:.3f}, "
                "Time {:.3f} (s)".format(
                    batch_idx, len(train_loader), loss, 0.0, iter_time))
            start_time = time.time()

        test(model, val_loader)


def main():
    args = parse_args()
    with open(args.config, 'r') as fid:
        config = json.load(fid)

    # seed everything:
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    # setup data loaders:
    dataset = config["data"]["dataset"]
    if not(hasattr(datasets, dataset)):
        raise ValueError(f"Unknown dataset {dataset}")
    dataset = getattr(datasets, dataset)

    input_size = config["data"]["img_height"]
    data_path = config["data"]["data_path"]
    preprocessor = dataset.Preprocessor(data_path, img_height=input_size)
    trainset = dataset.Dataset(data_path, preprocessor, split="train")
    valset = dataset.Dataset(data_path, preprocessor, split="validation")
    train_loader = utils.data_loader(trainset, config)
    val_loader = utils.data_loader(valset, config)

    # setup Model:
    output_size = preprocessor.num_classes + 1  # account for blank
    model = models.load_model(
        config["model"]["type"],
        input_size,
        output_size,
        **config["model"])
    model.to(device=args.device)
    criterion = models.ctc(blank=output_size - 1, pad_val=-1)

    # run training:
    train(
        model, criterion, train_loader, val_loader,
        epochs=config["optim"]["epochs"],
        lr=config["optim"]["learning_rate"],
        device=args.device)


if __name__ == "__main__":
    main()
