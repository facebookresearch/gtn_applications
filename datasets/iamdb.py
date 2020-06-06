import collections
import os
import PIL.Image
import random
import re
import torch
from torchvision import transforms


SPLITS = {
    "train" : ["trainset"],
    "validation" : ["validationset1"],
    "test" : ["validationset2", "testset"],
}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, preprocessor, split):
        forms = load_metadata(data_path)

        # Get split keys:
        splits = SPLITS.get(split, None)
        if splits is None:
            split_names = ", ".join(f"'{k}'" for k in SPLITS.keys())
            raise ValueError(
                f"Invalid split {split}, must be in [{split_names}].")

        split_keys = []
        for s in splits:
            with open(os.path.join(data_path, f"{s}.txt"), 'r') as fid:
                split_keys.extend((l.strip() for l in fid))

        self.preprocessor = preprocessor

        # setup image transforms:
        self.transforms = []
        if split == "train":
            self.transforms.extend([
                RandomResizeCrop(preprocessor.img_height),
                transforms.RandomRotation(2, fill=(256,)),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            ])
        else:
            self.transforms.append(ResizeCrop(preprocessor.img_height))

        self.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.912], std=[0.168]),
        ])
        self.transforms = transforms.Compose(self.transforms)

        # Load each image:
        self.dataset = []
        for key, lines in forms.items():
            for line in lines:
                if line["key"] not in split_keys:

                    continue
                img_file = os.path.join(data_path, f"{key}.png")
                self.dataset.append((img_file, line["box"], line["text"]))

    def sample_sizes(self):
        """
        Returns a list of tuples containing the input size
        (height, width) and the output length for each sample.
        """
        return (self.preprocessor.compute_size(box, line)
                for _, box, line in self.dataset)

    def __getitem__(self, index):
        img_file, box, text = self.dataset[index]
        img = PIL.Image.open(img_file)
        inputs = self.transforms((img, box))
        outputs = self.preprocessor.to_index(text)
        return inputs, outputs

    def __len__(self):
        return len(self.dataset)


class ResizeCrop:

    def __init__(self, img_height):
        self.img_height = img_height

    def __call__(self, args):
        img, box = args
        x, y, w, h = box
        size = (self.img_height, int((self.img_height / h) * w))
        return transforms.functional.resized_crop(
            img,
            y, x, h, w,
            size)


class RandomResizeCrop:

    def __init__(self, img_height, jitter=10, ratio=0.5):
        self.img_height = img_height
        self.jitter = jitter
        self.ratio = ratio

    def __call__(self, args):
        img, box = args
        # add some jitter to x, y, w, h:
        box = [b + random.randint(-self.jitter, self.jitter) for b in box]
        x, y, w, h = box

        # randomize aspect ratio:
        size_w = (self.img_height / h) * w
        size_w *= random.uniform(1 - self.ratio, 1 + self.ratio)
        size = (self.img_height, int(size_w))
        return transforms.functional.resized_crop(
            img,
            y, x, h, w,
            size)


class Preprocessor:

    def __init__(self, data_path, img_height):
        forms = load_metadata(data_path)

        # Build the token-to-index and index-to-token maps:
        tokens = set()
        for _, form in forms.items():
            for line in form:
                tokens.update(line["text"])
        self.index_to_tokens = sorted(list(tokens))
        self.tokens_to_index = { t : i
            for i, t in enumerate(self.index_to_tokens)}
        self.img_height = img_height

    def compute_size(self, box, line):
        x, y, w, h = box
        in_size = (self.img_height, int((self.img_height / h) * w))
        out_size = len(line)
        return in_size, out_size

    @property
    def num_classes(self):
        return len(self.index_to_tokens)

    def to_index(self, line):
        return torch.tensor([self.tokens_to_index[t] for t in line])

    def to_text(self, indices):
        return "".join(self.index_to_tokens[i] for i in indices)


def load_metadata(data_path):
    forms = collections.defaultdict(list)
    with open(os.path.join(data_path, "lines.txt"), 'r') as fid:
        lines = (l.strip().split() for l in fid if l[0] != "#")
        for line in lines:
            text = " ".join(line[8:])
            # remove garbage tokens:
            text = text.replace("#", "")
            text = re.sub(r"\|+", "|", text)
            form_key = "-".join(line[0].split("-")[:-1])
            forms[form_key].append({
                "key" : line[0],
                "box" : tuple(int(val) for val in line[4:8]),
                "text" : text,
            })

    return forms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute data stats.')
    parser.add_argument('--data_path', type=str,
        help='Path to dataset.')
    args = parser.parse_args()

    preprocessor = Preprocessor(args.data_path, 64)
    trainset = Dataset(args.data_path, preprocessor, split="train")
    valset = Dataset(args.data_path, preprocessor, split="validation")
    testset = Dataset(args.data_path, preprocessor, split="test")

    # Compute mean and var stats:
    images = torch.cat([trainset[i][0] for i in range(len(trainset))], dim=2)
    mean = torch.mean(images)
    std = torch.std(images)
    print(f"Data mean {mean} and standard deviation {std}.")
