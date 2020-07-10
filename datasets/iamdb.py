import collections
import multiprocessing as mp
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

    def __init__(self, data_path, preprocessor, split, augment=False):
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
        if augment:
            self.transforms.extend([
                RandomResizeCrop(),
                transforms.RandomRotation(2, fill=(255,)),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            ])
        self.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.912], std=[0.168]),
        ])
        self.transforms = transforms.Compose(self.transforms)

        # Load each image:
        images = []
        text = []
        for key, lines in forms.items():
            for line in lines:
                if line["key"] not in split_keys:
                    continue
                img_file = os.path.join(data_path, f"{key}.png")
                images.append((img_file, line["box"], preprocessor.img_height))
                text.append(line["text"])

        with mp.Pool(processes=16) as pool:
            images = pool.map(load_image, images)
        self.dataset = list(zip(images, text))

    def sample_sizes(self):
        """
        Returns a list of tuples containing the input size
        (width, height) and the output length for each sample.
        """
        return [(image.size, len(text)) for image, text in self.dataset]

    def __getitem__(self, index):
        img, text = self.dataset[index]
        inputs = self.transforms(img)
        outputs = self.preprocessor.to_index(text)
        return inputs, outputs

    def __len__(self):
        return len(self.dataset)


def load_image(example):
    img_file, box, height = example
    img = PIL.Image.open(img_file)
    x, y, w, h = box
    size = (height, int((height / h) * w))
    return transforms.functional.resized_crop(
        img,
        y, x, h, w,
        size)


class RandomResizeCrop:

    def __init__(self, jitter=10, ratio=0.5):
        self.jitter = jitter
        self.ratio = ratio

    def __call__(self, img):
        w, h = img.size

        # pad with white:
        img = transforms.functional.pad(img, self.jitter, fill=255)

        # crop at random (x, y):
        x = self.jitter + random.randint(-self.jitter, self.jitter)
        y = self.jitter + random.randint(-self.jitter, self.jitter)

        # randomize aspect ratio:
        size_w = w * random.uniform(1 - self.ratio, 1 + self.ratio)
        size = (h, int(size_w))
        img = transforms.functional.resized_crop(
            img,
            y, x, h, w,
            size)
        return img


class Preprocessor:

    def __init__(self, data_path, img_height, tokens_path=None):
        forms = load_metadata(data_path)

        # Load the set of graphemes:
        graphemes = set()
        for _, form in forms.items():
            for line in form:
                graphemes.update(line["text"])
        graphemes = sorted(list(graphemes))

        # Build the token-to-index and index-to-token maps:
        if tokens_path is not None:
            with open(tokens_path, 'r') as fid:
                tokens = sorted([l.strip() for l in fid])
        else:
            # Default to use graphemes if no tokens are provided
            tokens = graphemes

        self.tokens = tokens
        self.graphemes = graphemes
        self.graphemes_to_index = { t : i
            for i, t in enumerate(self.graphemes)}
        self.img_height = img_height

    @property
    def num_classes(self):
        return len(self.tokens)

    def to_index(self, line):
        return torch.tensor([self.graphemes_to_index[t] for t in line])

    def to_text(self, indices):
        # TODO should we use the tokens or the graphemes here?
        return "".join(self.graphemes[i] for i in indices)


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
    trainset = Dataset(
        args.data_path, preprocessor, split="train", augment=False)
    valset = Dataset(args.data_path, preprocessor, split="validation")
    testset = Dataset(args.data_path, preprocessor, split="test")

    # Compute mean and var stats:
    images = torch.cat([trainset[i][0] for i in range(len(trainset))], dim=2)
    mean = torch.mean(images)
    std = torch.std(images)
    print(f"Data mean {mean} and standard deviation {std}.")
