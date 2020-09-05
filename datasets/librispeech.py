import itertools
import json
import multiprocessing as mp
import os
import re
import torch
import torchaudio
import torchvision


SPLITS = {
    "train": ["train-clean-100"],
    "validation": ["dev-clean", "dev-other"],
    "test": ["test-clean", "test-other"],
}

WORDSEP = "▁"
SAMPLE_RATE = 16000


def log_transform(x):
    return torch.log(x + 1e-6)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, preprocessor, split, augment=False):
        data = []
        for sp in SPLITS[split]:
            data.extend(load_data_split(data_path, sp))

        self.preprocessor = preprocessor

        # setup transforms:
        self.transforms = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=preprocessor.img_height
            ),
            torchvision.transforms.Lambda(log_transform),
            torchvision.transforms.Normalize(mean=[-5.532], std=[4.02]),
        ]
        if augment:
            self.transforms.extend(
                [
                    torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                    torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                    torchaudio.transforms.TimeMasking(100, iid_masks=True),
                    torchaudio.transforms.TimeMasking(100, iid_masks=True),
                ]
            )
        self.transforms = torchvision.transforms.Compose(self.transforms)

        # Load each audio file:
        audio = [example["audio"] for example in data]
        text = [example["text"] for example in data]
        duration = [example["duration"] for example in data]
        self.dataset = list(zip(audio, text, duration))

    def sample_sizes(self):
        """
        Returns a list of tuples containing the input size
        (time, 1) and the output length for each sample.
        """
        return [((duration, 1), len(text)) for _, text, duration in self.dataset]

    def __getitem__(self, index):
        audio_file, text, _ = self.dataset[index]
        audio = torchaudio.load(audio_file)
        inputs = self.transforms(audio[0])
        outputs = self.preprocessor.to_index(text)
        return inputs, outputs

    def __len__(self):
        return len(self.dataset)


class Preprocessor:
    """
    A preprocessor for the Librispeech dataset.
    Args:
        data_path (str) : Path to the top level data directory.
        img_height (int) : Number of audio features in transform.
        tokens_path (str) (optional) : The path to the list of model output
            tokens. If not provided the token set is built dynamically from
            the graphemes of the tokenized text. NB: This argument does not
            affect the tokenization of the text, only the number of output
            classes.
        lexicon_path (str) (optional) : A mapping of words to tokens. If
            provided the preprocessor will split the text into words and
            map them to the corresponding token. If not provided the text
            will be tokenized at the grapheme level.
    """

    def __init__(
        self,
        data_path,
        img_height,
        tokens_path=None,
        lexicon_path=None,
        use_words=False,
        prepend_wordsep=False,
    ):
        if use_words:
            raise ValueError("use_words not supported for Librispeech dataset")
        self._prepend_wordsep = prepend_wordsep
        self.img_height = img_height

        data = []
        for sp in SPLITS["train"]:
            data.extend(load_data_split(data_path, sp))

        # Load the set of graphemes:
        graphemes = set()
        for ex in data:
            graphemes.update(ex["text"])
        self.graphemes = sorted(graphemes)

        # Build the token-to-index and index-to-token maps:
        if tokens_path is not None:
            with open(tokens_path, "r") as fid:
                self.tokens = sorted([l.strip() for l in fid])
        else:
            # Default to use graphemes if no tokens are provided
            self.tokens = self.graphemes

        if lexicon_path is not None:
            with open(lexicon_path, "r") as fid:
                lexicon = (l.strip().split() for l in fid)
                lexicon = {l[0]: l[1:] for l in lexicon}
                self.lexicon = lexicon
        else:
            self.lexicon = None

        self.graphemes_to_index = {t: i for i, t in enumerate(self.graphemes)}
        self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}

    @property
    def num_tokens(self):
        return len(self.tokens)

    def to_index(self, line):
        tok_to_idx = self.graphemes_to_index
        if self.lexicon is not None:
            if len(line) > 0:
                # If the word is not found in the lexicon, fall back to letters.
                line = [
                    t
                    for w in line.split(WORDSEP)
                    for t in self.lexicon.get(w, WORDSEP + w)
                ]
            tok_to_idx = self.tokens_to_index
        # In some cases we require the target to start with WORDSEP, for
        # example when learning word piece decompositions.
        if self._prepend_wordsep:
            line = itertools.chain([WORDSEP], line)
        return torch.LongTensor([tok_to_idx[t] for t in line])

    def to_text(self, indices):
        # Roughly the inverse of `to_index`
        encoding = self.graphemes
        if self.lexicon is not None:
            encoding = self.tokens
        return self._post_process(encoding[i] for i in indices)

    def tokens_to_text(self, indices):
        return self._post_process(self.tokens[i] for i in indices)

    def _post_process(self, indices):
        # ignore preceding and trailling spaces
        return "".join(indices).strip(WORDSEP)


def load_data_split(data_path, split):
    json_file = os.path.join(data_path, f"{split}.json")
    with open(json_file, "r") as fid:
        examples = [json.loads(l) for l in fid]
        for ex in examples:
            text = ex["text"]
            # swap word sep from | to WORDSEP
            text = re.sub(r"\s", WORDSEP, text).strip(WORDSEP)
            ex["text"] = text
    return examples


if __name__ == "__main__":
    import argparse

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

    preprocessor = Preprocessor(args.data_path, 80)
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
