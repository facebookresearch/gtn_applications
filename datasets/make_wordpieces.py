import argparse
import io
import logging
import os
import sentencepiece as spm


def iamdb_pieces(args):
    import iamdb
    forms = iamdb.load_metadata(args.data_dir)
    with open(os.path.join(args.data_dir, "trainset.txt"), 'r') as fid:
        train = [l.strip() for l in fid]

    text = []
    for key, lines in forms.items():
        for line in lines:
            if line["key"] not in train:
                continue
            text.append(line["text"])
    sp = train_spm_model((t.replace("|", " ") for t in text), args.num_pieces)
    pieces = [sp.id_to_piece(i) for i in range(sp.vocab_size())]
    # save pieces to a file
    with open(args.output, 'w') as fid:
        fid.write("\n".join(pieces))


def train_spm_model(sentences, vocab_size):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_writer=model,
        vocab_size=vocab_size,
        character_coverage=1.0,
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
    parser.add_argument("--output",
         default="word_pieces.txt",
         type=str,
         help="List of word pieces.",
     )
    parser.add_argument("--num_pieces",
         default=1000,
         type=int,
         help="Number of word pieces.",
     )
    args = parser.parse_args()

    if args.dataset == "iamdb":
        logging.info("Building word pieces for IAMDB")
        iamdb_pieces(args)
