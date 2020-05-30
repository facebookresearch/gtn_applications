import collections
import os
import PIL.Image
import torch
import torchvision


SPLITS = {
    "train" : ["trainset"],
    "validation" : ["validationset1", "validationset2"],
    "test" : ["testset"],
}


class IamDB(torch.utils.data.Dataset):

    def __init__(self, data_path, preprocessor, split):
        forms = load_metadata(data_path)

        # Get split keys:
        splits = SPLITS.get(split, None)
        if splits is None:
            split_names = ", ".join(f"'{k}'" for k in SPLITS.keys())
            raise ValueError(
                f"Invalid split {split}, must be in [{split_names}].")

        split_keys = []
        for split in splits:
            with open(os.path.join(data_path, f"{split}.txt"), 'r') as fid:
                split_keys.extend((l.strip() for l in fid))

        self.preprocessor = preprocessor

        # Load each image:
        self.dataset = []
        for key, lines in forms.items():
            if lines[0]["key"] not in split_keys:
                continue
            img = PIL.Image.open(os.path.join(data_path, f"{key}.png"))
            for line in lines:
                if line["key"] not in split_keys:
                    continue
                self.dataset.append((img, line["box"], line["text"]))

    def __getitem__(self, index):
        img, box, text = self.dataset[index]
        x, y, w, h = box
        size = (20, int((20 / h) * w))
        inputs = torchvision.transforms.functional.resized_crop(
            img,
            y, x, h, w,
            size)
        outputs = self.preprocessor.to_index(text)
        return inputs, outputs

    def __len__(self):
        return len(self.dataset)


class Preprocessor:

    def __init__(self, data_path):
        forms = load_metadata(data_path)

        # Build the token-to-index and index-to-token maps:
        tokens = set()
        for _, form in forms.items():
            for line in form:
                tokens.update(line["text"])
        self.index_to_tokens = sorted(list(tokens))
        self.tokens_to_index = { t : i
            for i, t in enumerate(self.index_to_tokens)}

    def to_index(self, line):
        return [self.tokens_to_index[t] for t in line]

    def to_text(self, indices):
        return "".join(self.index_to_tokens[i] for i in indices)


def load_metadata(data_path):
    forms = collections.defaultdict(list)
    with open(os.path.join(data_path, "lines.txt"), 'r') as fid:
        lines = (l.strip().split() for l in fid if l[0] != "#")
        for line in lines:
            form_key = "-".join(line[0].split("-")[:-1])
            forms[form_key].append({
                "key" : line[0],
                "box" : tuple(int(val) for val in line[4:8]),
                "text" : line[-1],
            })
    return forms


if __name__ == "__main__":
    data_path = "data/"
    preprocessor = Preprocessor()
    trainset = IamDB(data_path, preprocessor, split="train")
    valset = IamDB(data_path, preprocessor, split="validation")
    testset = IamDB(data_path, preprocessor, split="test")
