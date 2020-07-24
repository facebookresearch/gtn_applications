import argparse
import editdistance
import json
import logging
import os
import torch

import datasets
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalaute a handwriting recognition model.")
    parser.add_argument("--config", type=str,
        help="The json configuration file used for training."
    )
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument(
        "--checkpoint_path",
        default="/tmp/",
        type=str,
        help="Checkpoint path for loading the model",
    )
    parser.add_argument("--load_last", default=False, action='store_true',
        help="Load the last saved model instead of the best model.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        type=str,
        choices=['train', 'validation', 'test'],
        help="Data split to test on (default: 'validation')",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return args


def load(model, criterion, checkpoint_path, load_last=False):
    model_checkpoint = os.path.join(checkpoint_path, "model.checkpoint")
    criterion_checkpoint = os.path.join(checkpoint_path, "criterion.checkpoint")
    if not load_last:
        model_checkpoint += ".best"
        criterion_checkpoint += ".best"
    model.load_state_dict(torch.load(model_checkpoint))
    criterion.load_state_dict(torch.load(criterion_checkpoint))


@torch.no_grad()
def test(args):
    with open(args.config, "r") as fid:
        config = json.load(fid)

    if not args.disable_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = config["data"]["dataset"]
    if not (hasattr(datasets, dataset)):
        raise ValueError(f"Unknown dataset {dataset}")
    dataset = getattr(datasets, dataset)

    input_size = config["data"]["img_height"]
    data_path = config["data"]["data_path"]
    preprocessor = dataset.Preprocessor(
            data_path,
            img_height=input_size,
            tokens_path=config["data"].get("tokens", None),
            lexicon_path=config["data"].get("lexicon", None))
    data = dataset.Dataset(data_path, preprocessor, split=args.split)
    loader = utils.data_loader(data, config)

    criterion, output_size = models.load_criterion(
        config.get("criterion_type", "ctc"),
        preprocessor,
        config.get("criterion", {}),
    )
    criterion = criterion.to(device)
    model = models.load_model(
        config["model_type"], input_size, output_size, config["model"]
    ).to(device)
    load(model, criterion, args.checkpoint_path, args.load_last)

    model.eval()
    meters = utils.Meters()
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        meters.loss += criterion(outputs, targets).item() * len(targets)
        meters.num_samples += len(targets)
        predictions = criterion.viterbi(outputs)
        for p, t in zip(predictions, targets):
            p, t = preprocessor.tokens_to_text(p), preprocessor.to_text(t)
            dist = editdistance.eval(p, t)
            print("CER: {:.3f}".format(dist / len(t)))
            print("HYP:", "".join(p))
            print("REF", "".join(t))
            print("="*80)
            meters.edit_distance += dist
            meters.num_tokens += len(t)

    logging.info("Loss {:.3f}, CER {:.3f}, ".format(
        meters.avg_loss, meters.cer))


def main():
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
