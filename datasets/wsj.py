"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torchaudio

import audioset


class Dataset(audioset.Dataset):

    splits = {
        "train": ["train_si284"],
        "validation": ["dev_93"],
        "test": ["eval_92"],
    }

    sample_rate = 16000

    def __init__(self, data_path, preprocessor, split, augment=False):
        augmentation = []
        if augment:
            augmentation = [
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
            ]

        super(Dataset, self).__init__(
            data_path,
            preprocessor,
            split,
            self.splits,
            augmentation=augmentation,
            sample_rate=self.sample_rate,
        )


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Compute data stats.")
    parser.add_argument("--data_path", type=str, help="Path to dataset JSON files.")
    parser.add_argument(
        "--save_text", type=str, help="Path to save parsed train text.", default=None
    )
    parser.add_argument(
        "--save_tokens", type=str, help="Path to save tokens.", default=None
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Compute training data statistics.",
        default=False,
    )
    args = parser.parse_args()

    preprocessor = audioset.Preprocessor(args.data_path, 80)
    print(f"Number of tokens: {preprocessor.num_tokens}")
    trainset = Dataset(args.data_path, preprocessor, split="train", augment=False)
    if args.save_text is not None:
        with open(args.save_text, "w") as fid:
            fid.write("\n".join(t for _, t, _ in trainset.dataset))
    if args.save_tokens is not None:
        with open(args.save_tokens, "w") as fid:
            fid.write("\n".join(preprocessor.tokens))
    valset = Dataset(args.data_path, preprocessor, split="validation")
    testset = Dataset(args.data_path, preprocessor, split="test")
    print("Number of examples per dataset:")
    print(f"Training: {len(trainset)}")
    print(f"Validation: {len(valset)}")
    print(f"Test: {len(testset)}")

    if not args.compute_stats:
        import sys

        sys.exit(0)

    # Compute mean and var stats:
    audio = torch.cat([trainset[i][0] for i in range(len(trainset))], dim=2)
    mean = torch.mean(audio)
    std = torch.std(audio)
    print(f"Data mean {mean} and standard deviation {std}.")

    # Compute average lengths of audio and targets:
    avg_in_t = sum(w for (w, _), _ in trainset.sample_sizes()) / len(trainset)
    avg_tgt_l = sum(l for _, l in trainset.sample_sizes()) / len(trainset)
    print(f"Average audio length {avg_in_t} (s)")
    print(f"Average target length {avg_tgt_l}")
