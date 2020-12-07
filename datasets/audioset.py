import itertools
import json
import os
import re
import torch
import torchaudio
import torchvision


def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, preprocessor, split, splits, augmentation=None, sample_rate=16000):
        data = []
        for sp in splits[split]:
            data.extend(load_data_split(data_path, sp, preprocessor.wordsep))

        self.preprocessor = preprocessor

        # setup transforms:
        self.transforms = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, sample_rate * 25 // 1000,
                n_mels=preprocessor.num_features,
                hop_length=sample_rate * 10 // 1000,
            ),
            torchvision.transforms.Lambda(log_normalize),
        ]
        if augmentation is not None:
            self.transforms.extend(augmentation)
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
    A preprocessor for an audio dataset.
    Args:
        data_path (str) : Path to the top level data directory.
        num_features (int) : Number of audio features in transform.
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
        num_features,
        splits,
        tokens_path=None,
        lexicon_path=None,
        use_words=False,
        prepend_wordsep=False,
    ):
        if use_words:
            raise ValueError("use_words not supported for audio dataset")
        self.wordsep = "▁"
        self._prepend_wordsep = prepend_wordsep
        self.num_features = num_features

        data = []
        for sp in splits["train"]:
            data.extend(load_data_split(data_path, sp, self.wordsep))

        # Load the set of graphemes:
        graphemes = set()
        for ex in data:
            graphemes.update(ex["text"])
        self.graphemes = sorted(graphemes)

        # Build the token-to-index and index-to-token maps:
        if tokens_path is not None:
            with open(tokens_path, "r") as fid:
                self.tokens = [l.strip() for l in fid]
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
                    for w in line.split(self.wordsep)
                    for t in self.lexicon.get(w, self.wordsep + w)
                ]
            tok_to_idx = self.tokens_to_index
        # In some cases we require the target to start with self.wordsep, for
        # example when learning word piece decompositions.
        if self._prepend_wordsep:
            line = itertools.chain([self.wordsep], line)
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
        return "".join(indices).strip(self.wordsep)


def load_data_split(data_path, split, wordsep):
    json_file = os.path.join(data_path, f"{split}.json")
    with open(json_file, "r") as fid:
        examples = [json.loads(l) for l in fid]
        for ex in examples:
            text = ex["text"]
            # swap word sep from | to self.wordsep
            text = re.sub(r"\s", wordsep, text).strip(wordsep)
            ex["text"] = text
    return examples
