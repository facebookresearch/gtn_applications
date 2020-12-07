"""
Prepare the WSJ dataset.

To convert all the sphere files (.wv1) into wave files use `--convert`.  This
requires installing `sph2pipe`, which can be done with `./install_sph2pipe.sh`.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import torchaudio

DATASETS = {
    "train_si284" : [
        "csr_2_comp/13-34.1/wsj1/doc/indices/si_tr_s.ndx",
        "csr_1/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx"
    ],
    "eval_92" : [
        "csr_1/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx"
    ],
    "dev_93" : [
        "csr_2_comp/13-34.1/wsj1/doc/indices/h1_p0.ndx"
    ]
}

DOT_PATHS = [
    "csr_1/*/wsj0/transcrp/dots/*/*/*.dot",
    "csr_2_comp/13-34.1/wsj1/trans/wsj1/*/*/*.dot",
    "csr_1/11-14.1/wsj0/si_et_20/*/*.dot"
]

ALLOWED = set("abcdefghijklmnopqrstuvwxyz.' -")

REPLACE = {
    ".point" : "point",
    ".period": "period",
    "'single-quote": "single-quote",
    "'single-close-quote": "single-close-quote",
    "`single-quote" : "single-quote",
    "-hyphen": "hyphen",
    ")close_paren" : "close-paren",
    "(left(-paren)-": "left-",
    "." : "",
    "--dash" : "dash",
    "-dash" : "dash",
}


def load_text(wsj_base):
    transcripts = []
    dots = []
    for d in DOT_PATHS:
        dots.extend(glob.glob(os.path.join(wsj_base, d)))
    for f in dots:
        with open(f, 'r') as fid:
            transcripts.extend(l.strip() for l in fid)
    transcripts = (t.split() for t in transcripts)
    # Key text by utterance id
    transcripts = {t[-1][1:-1] : clean(" ".join(t[:-1]))
                    for t in transcripts}
    return transcripts


def load_waves(wsj_base, files):
    waves = []

    def to_disk(d):
        return "{}-{}.{}".format(*d.split("_"))

    for f in files:
        disk = f.split(os.sep)[0]
        flist = os.path.join(wsj_base, f)
        with open(flist, 'r') as fid:
            lines = (l.split(":") for l in fid if l[0] != ';')
            lines = (os.path.join(to_disk(k1), k2.strip().strip("/"))
                for k1, k2 in lines)
            lines = (os.path.join(wsj_base, disk, l) for l in lines)
            waves.extend(sorted(lines))
    return waves


def clean(line):
    pl = line
    line = line.lower()
    line = re.sub("<|>|\\\\|\[\S+\]", "", line)
    toks = line.split()
    clean_toks = []
    for tok in toks:
        if re.match("\S+-dash", tok):
            clean_toks.extend(tok.split("-"))
        else:
            clean_toks.append(REPLACE.get(tok, tok))
    line = " ".join(t for t in clean_toks if t).strip()
    line = re.sub("\(\S*\)", "", line)
    line = re.sub("[()\*\":\?;!}{\~<>/&,\$\%\~]", "", line)
    line = re.sub("`", "'", line)
    line = " ".join(line.split())
    return line


def write_json(save_path, dataset, waves, transcripts):
    out_file = os.path.join(save_path, dataset + ".json")
    with open(out_file, 'w') as fid:
        for wave_file in waves:
            audio = torchaudio.load(wave_file)
            duration = audio[0].numel() / audio[1]
            key = os.path.basename(wave_file)
            key = os.path.splitext(key)[0]
            datum = {'text' : transcripts[key],
                     'duration' : duration,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")


def convert_sph_to_wav(files, out_path):
    command = ["sph2pipe_v2.5/sph2pipe", "-p", "-f",
               "wav", "-c", "1"]
    converted = []
    for sph_f in files:
        f, ext = os.path.splitext(os.path.basename(sph_f))
        if ext == "":
            sph_f = "{}.wv1".format(sph_f)
        out_f = f + ".wav"
        out_f = os.path.join(out_path, out_f)
        converted.append(out_f)
        subprocess.call(command + [sph_f, out_f])
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WSJ dataset.")
    parser.add_argument("--data_path",
        help="Location of WSJ root directory.")
    parser.add_argument("--save_path",
        help="Path to save dataset jsons.", default=".")
    parser.add_argument("--convert", action="store_true",
        help="Convert sphere to wav format.")
    args = parser.parse_args()

    transcripts = load_text(args.data_path)
    for d, v in DATASETS.items():
        waves = load_waves(args.data_path, v)
        out_path = os.path.abspath(os.path.join(args.save_path, d))
        os.mkdir(out_path)
        if d == "train_si284":
            waves = filter(lambda x: "wsj0/si_tr_s/401" not in x, waves)
        if args.convert:
            print("Converting {}".format(d))
            waves = convert_sph_to_wav(waves, out_path)
        print("Writing {}".format(d))
        write_json(args.save_path, d, waves, transcripts)

