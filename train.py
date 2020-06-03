import argparse
import json
import logging
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
    logging.basicConfig(level=logging.INFO)
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    logging.info(f"Training with {args.device}.")
    return args


def compute_edit_distance(predictions, targets):
    dist = 0
    n_tokens = 0
    for p, t in zip(predictions, targets):
        dist += utils.edit_distance(p, t)[0].item()
        n_tokens += t.numel()
    return dist, n_tokens


def test(model, criterion, data_loader, device):
    model.eval()
    loss = 0.0
    n = 0
    distance = 0
    n_tokens = 0
    for inputs, targets in data_loader:
        outputs = model(inputs.to(device))
        loss += criterion(outputs, targets).item() * len(targets)
        n += len(targets)
        dist, toks = compute_edit_distance(criterion.decode(outputs), targets)
        distance += dist
        n_tokens += toks

    logging.info("Loss {:.3f}, CER {:.3f}".format(
        len(data_loader), loss / n, distance / n_tokens))


def train(
        model, criterion, train_loader, valid_loader,
        epochs, lr, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            dist, tot = compute_edit_distance(criterion.decode(outputs), targets)
            iter_time = time.time() - start_time
            logging.info(
                "Batch {}/{}: Loss {:.3f}, CER {:.3f}, Time {:.3f} (s)".format(
                    batch_idx, len(train_loader), loss, dist / tot, iter_time))
            start_time = time.time()

        print("Evaluating validation set...")
        test(model, criterion, valid_loader, device)


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
    criterion = models.CTC(blank=output_size - 1)

    # run training:
    train(
        model, criterion, train_loader, val_loader,
        epochs=config["optim"]["epochs"],
        lr=config["optim"]["learning_rate"],
        device=args.device)


if __name__ == "__main__":
    main()
