import argparse
import io
import os
import sentencepiece as spm


def iamdb_pieces(args):
    import iamdb
    forms = iamdb.load_metadata(args.data_dir)
    with open(os.path.join(args.data_dir, "trainset.txt"), 'r') as fid:
        train = [l.strip() for l in fid]

    # Train sentencepiece model only on the training set
    all_lines = [l for _, lines in forms.items() for l in lines]
    text = [l["text"] for l in all_lines]
    train_text = [l["text"] for l in all_lines if l["key"] not in train]
    num_pieces = args.num_pieces
    sp = train_spm_model(
        (t.replace("|", " ") for t in train_text),
        num_pieces + 1, # to account for <unk>
        user_symbols=["/"])
    vocab = sorted(set(
        w for t in text for w in t.replace("|", " ").split(" ") if w))
    print(f"Generating word piece list of size {num_pieces}.")
    pieces = [sp.id_to_piece(i) for i in range(1, num_pieces + 1)]
    print(f"Encoding vocabulary of size {len(vocab)}.")
    encoded_vocab = [sp.encode_as_pieces(v) for v in vocab]

    # save pieces to a file
    with open(args.output_prefix + f"_tokens_{num_pieces}.txt", 'w') as fid:
        fid.write("\n".join(pieces))
    # save lexicon to a file
    with open(args.output_prefix + f"_lex_{num_pieces}.txt", 'w') as fid:
        for v, p in zip(vocab, encoded_vocab):
            fid.write("{} {}\n".format(v, " ".join(p)))


def train_spm_model(sentences, vocab_size, user_symbols=None):
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
        description="Make the word piece list for a datset.")
    parser.add_argument(
        "--dataset",
        default="iamdb",
        type=str,
        help="Name of the dataset.",
        choices=["iamdb"],
    )
    parser.add_argument(
        "--data_dir",
        default="/datasets01/iamdb/060820/",
        type=str,
        help="Name of the dataset.",
    )
    parser.add_argument("--output_prefix",
         default="word_pieces",
         type=str,
         help="Prefix path/name to store tokens and lexicon.",
     )
    parser.add_argument("--num_pieces",
         default=1000,
         type=int,
         help="Number of word pieces.",
     )
    args = parser.parse_args()

    if args.dataset == "iamdb":
        print("Building word pieces for IAMDB")
        iamdb_pieces(args)
