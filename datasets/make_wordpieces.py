import argparse
import io
import os
import sentencepiece as spm


def iamdb_pieces(args):
    import iamdb

    forms = iamdb.load_metadata(args.data_dir)
    with open(os.path.join(args.data_dir, "trainset.txt"), "r") as fid:
        train = [l.strip() for l in fid]

    # Train sentencepiece model only on the training set
    text = [l for _, lines in forms.items() for l in lines if l["key"] in train]
    num_pieces = args.num_pieces
    sp = train_spm_model(
        iter(train_text),
        num_pieces + 1,  # to account for <unk>
        user_symbols=["/"],  # added so token is in the output set
    )
    vocab = sorted(set(w for t in text for w in t.split("▁") if w))
    save_pieces(sp, num_pieces, args.output_prefix, vocab)


def librispeech_pieces(args):
    import librispeech

    # Train sentencepiece model only on the training set
    text = []
    for subset in librispeech.SPLITS["train"]:
        ds = librispeech.load_data_split(args.data_dir, subset)
        text.extend(l["text"] for l in ds)
    num_pieces = args.num_pieces
    sp = train_spm_model(
        iter(text),
        num_pieces + 1,  # to account for <unk>
    )
    vocab = sorted(set(w for t in text for w in t.split("▁") if w))
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
        choices=["iamdb", "librispeech"],
    )
    parser.add_argument(
        "--data_dir",
        default="/datasets01/iamdb/060820/",
        type=str,
        help="Path to the dataset.",
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
