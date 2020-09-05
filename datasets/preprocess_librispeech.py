import argparse
import glob
import json
import os
import torchaudio


SPLITS = [
    "train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other",
]


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip().split() for l in fid)
            lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data


def path_from_key(key, prefix, ext):
    dirs = key.split("-")
    dirs[-1] = key
    path = os.path.join(prefix, *dirs)
    return path + os.path.extsep + ext


def clean_text(text):
    return text.strip().lower()


def build_json(data_path, save_path, split):
    split_path = os.path.join(data_path, split)
    transcripts = load_transcripts(split_path)
    save_file = os.path.join(save_path, f"{split}.json")
    with open(save_file, 'w') as fid:
        for k, t in transcripts.items():
            flac_file = path_from_key(k, split_path, ext="flac")
            audio = torchaudio.load(flac_file)
            duration = audio[0].numel() / audio[1]
            t = clean_text(t)
            datum = {'text' : t,
                     'duration' : duration,
                     'audio' : flac_file}
            json.dump(datum, fid)
            fid.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("--data_path", type=str,
        help="Location of the librispeech root directory.")
    parser.add_argument("--save_path", type=str,
        help="The json is saved in <save_path>/{train-clean-100, ...}.json")
    args = parser.parse_args()

    for split in SPLITS:
        print("Preprocessing {}".format(split))
        build_json(args.data_path, args.save_path, split)
