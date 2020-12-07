import argparse
import io
import os
import sentencepiece as spm


def iamdb_pieces(args):
    import iamdb
    forms = iamdb.load_metadata(args.data_dir)
    ds_keys = set()
    for _, v in iamdb.SPLITS.items():
        for ds in v:
            with open(os.path.join(args.data_dir, f"{ds}.txt"), "r") as fid:
                ds_keys.update(l.strip() for l in fid)

    # Train sentencepiece model only on the training set
    text = [l["text"] for _, lines in forms.items()
        for l in lines if l["key"] not in ds_keys]
    num_pieces = args.num_pieces
    sp = train_spm_model(
        iter(text),
        num_pieces + 1,  # to account for <unk>
        user_symbols=["/"],  # added so token is in the output set
    )
    vocab = sorted(set(w for t in text for w in t.split("▁") if w))
    save_pieces(sp, num_pieces, args.output_prefix, vocab)


def librispeech_pieces(args):
    # Train sentencepiece model only on the training set
    import librispeech
    json_set_pieces(args, librispeech)


def wsj_pieces(args):
    import wsj
    # Load the 20k open vocabulary:
    # Expects the original 20k vocab to be copied from
    # "csr_2_comp/13-34.1/wsj1/doc/lng_modl/vocab/wlist20o.nvp"
    # to "<data_dir>/vocab20ko.txt"
    vocab_file = os.path.join(args.data_dir, "vocab20ko.txt")
    with open(vocab_file, 'r') as fid:
        vocab = [l.strip().lower() for l in fid if l[0] != "#"]
    json_set_pieces(args, wsj, vocab)


def json_set_pieces(args, dataset, vocab=None):
    # Train sentencepiece model only on the training set
    train_text = []
    for subset in dataset.SPLITS["train"]:
        ds = dataset.load_data_split(args.data_dir, subset)
        train_text.extend(l["text"] for l in ds)
    if args.text_file is not None:
        with open(args.text_file, "r") as fid:
            spm_text = [l.strip() for l in fid]
    else:
        spm_text = train_text
    num_pieces = args.num_pieces
    sp = train_spm_model(
        iter(spm_text),
        num_pieces + 1,  # to account for <unk>
    )
    if vocab is None:
        vocab = sorted(set(w for t in train_text for w in t.split("▁") if w))
    save_pieces(sp, num_pieces, args.output_prefix, vocab)


def save_pieces(sp, num_pieces, output_prefix, vocab):
    print(f"Generating word piece list of size {num_pieces}.")
    pieces = [sp.id_to_piece(i) for i in range(1, num_pieces + 1)]
    print(f"Encoding vocabulary of size {len(vocab)}.")
    encoded_vocab = [sp.encode_as_pieces(v) for v in vocab]

    # save pieces to a file
    with open(output_prefix + f"_tokens_{num_pieces}.txt", "w") as fid:
        fid.write("\n".join(pieces))
    # save lexicon to a file
    with open(output_prefix + f"_lex_{num_pieces}.txt", "w") as fid:
        for v, p in zip(vocab, encoded_vocab):
            fid.write("{} {}\n".format(v, " ".join(p)))


def train_spm_model(sentences, vocab_size, user_symbols=""):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_writer=model,
        vocab_size=vocab_size,
        bos_id=-1,
        eos_id=-1,
        character_coverage=1.0,
        user_defined_symbols=user_symbols,
    )
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    return sp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make the word piece list for a datset."
    )
    parser.add_argument(
        "--dataset",
        default="iamdb",
        type=str,
        help="Name of the dataset.",
        choices=["iamdb", "librispeech", "wsj"],
    )
    parser.add_argument(
        "--data_dir",
        default="/datasets01/iamdb/060820/",
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--text_file",
        default=None,
        type=str,
        help="Path to sentence piece training text",
    )
    parser.add_argument(
        "--output_prefix",
        default="word_pieces",
        type=str,
        help="Prefix path/name to store tokens and lexicon.",
    )
    parser.add_argument(
        "--num_pieces",
        default=1000,
        type=int,
        help="Number of word pieces.",
    )
    args = parser.parse_args()

    print(f"Building word pieces for {args.dataset}")
    locals()[args.dataset + "_pieces"](args)
