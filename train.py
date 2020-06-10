import argparse
from dataclasses import dataclass
import editdistance
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
        dist += editdistance.eval(p.tolist(), t.tolist())
        n_tokens += t.numel()
    return dist, n_tokens


@dataclass
class Meters:
    loss = 0.0
    num_samples = 0
    num_tokens = 0
    edit_distance = 0

    @property
    def avg_loss(self):
        return self.loss / self.num_samples

    @property
    def cer(self):
        return self.edit_distance / self.num_tokens


@torch.no_grad()
def test(model, criterion, data_loader, device):
    model.eval()
    meters = Meters()
    for inputs, targets in data_loader:
        outputs = model(inputs.to(device))
        meters.loss += criterion(outputs, targets).item() * len(targets)
        meters.num_samples += len(targets)
        dist, toks = compute_edit_distance(
            criterion.decode(outputs), targets)
        meters.edit_distance += dist
        meters.num_tokens += toks

    return meters.avg_loss, meters.cer


def train(
        model, criterion, train_loader, valid_loader,
        epochs, lr, device, step_size):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.5)

    min_val_loss = float("inf")
    min_val_cer = float("inf")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        meters = Meters()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            meters.loss += loss.item() * len(targets)
            meters.num_samples += len(targets)
            dist, toks = compute_edit_distance(criterion.decode(outputs), targets)
            meters.edit_distance += dist
            meters.num_tokens += toks
        epoch_time = time.time() - start_time
        logging.info(
            "Epoch {} complete. "
            "Loss {:.3f}, CER {:.3f}, Time {:.3f} (s)".format(
                epoch + 1, meters.avg_loss, meters.cer, epoch_time))
        logging.info("Evaluating validation set..")
        val_loss, val_cer = test(model, criterion, valid_loader, device)
        min_val_loss = min(val_loss, min_val_loss)
        min_val_cer = min(val_cer, min_val_cer)
        logging.info("Validation Set: Loss {:.3f}, CER {:.3f}, "
            "Best Loss {:.3f}, Best CER {:.3f}".format(
            val_loss, val_cer, min_val_loss, min_val_cer))

        scheduler.step()
        start_time = time.time()


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
    trainset = dataset.Dataset(
        data_path, preprocessor, split="train", augment=True)
    valset = dataset.Dataset(data_path, preprocessor, split="validation")
    train_loader = utils.data_loader(trainset, config)
    val_loader = utils.data_loader(valset, config)

    # setup Model:
    output_size = preprocessor.num_classes + 1  # account for blank
    model = models.load_model(
        config["model_type"],
        input_size,
        output_size,
        config["model"])
    model.to(device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Training {} model with {:,} parameters.".format(
        config["model_type"], n_params))
    criterion = models.CTC(blank=output_size - 1)

    # run training:
    train(
        model, criterion, train_loader, val_loader,
        epochs=config["optim"]["epochs"],
        lr=config["optim"]["learning_rate"],
        device=args.device,
        step_size=config["optim"]["step_size"])


if __name__ == "__main__":
    main()
