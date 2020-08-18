import collections
import itertools
import gtn

START_IDX = -1
WORDSEP = "▁"


def build_graph(ngrams):
    graph = gtn.Graph(False)
    ngram = len(ngrams)
    state_to_node = {}
    start_state = tuple([START_IDX] * (ngram - 1))

    def get_node(state):
        node = state_to_node.get(state, None)
        if node is not None:
            return node
        start = state == start_state
        node = graph.add_node(start, True)
        state_to_node[state] = node
        # Add back off when adding node
        for n in range(1, len(state) + 1):
            back_off_node = state_to_node.get(state[n:], None)
            # Epsilon transition to the back-off state
            if back_off_node is not None:
                graph.add_arc(node, back_off_node, gtn.epsilon)
        return node

    for grams in ngrams:
        for gram in grams:
            istate, ostate = gram[0:-1], gram[len(gram) - ngram + 1 :]
            inode = get_node(istate)
            if gram[1:] not in state_to_node:
                raise ValueError(
                    "Ill formed counts: if (x, y_1, ..., y_{n-1}) is above"
                    "the n-gram threshold, then (y_1, ..., y_{n-1}) must be"
                    "above the (n-1)-gram threshold"
                )
            onode = get_node(ostate)
            # p(gram[-1] | gram[:-1])
            graph.add_arc(inode, onode, gram[-1])
    return graph


def count_ngrams(lines, ngram, tokens_to_idx):
    counts = [collections.Counter() for _ in range(ngram)]
    for line in lines:
        # prepend implicit start token
        token_line = [START_IDX] * (ngram - 1)
        for t in line:
            token_line.append(tokens_to_idx[t])
        for e in range(ngram - 1, len(token_line)):
            for n, counter in enumerate(counts):
                counter[tuple(token_line[e - n : e + 1])] += 1
    return counts


def prune_ngrams(ngrams, prune):
    pruned_ngrams = []
    for n, grams in enumerate(ngrams):
        grams = grams.most_common()
        pruned_grams = [gram for gram, c in grams if c > prune[n]]
        pruned_ngrams.append(pruned_grams)
    return pruned_ngrams


def add_blank_grams(pruned_ngrams, num_tokens, blank):
    all_grams = [gram for grams in pruned_ngrams for gram in grams]
    maxorder = len(pruned_ngrams)
    blank_grams = {}
    if blank == "forced":
        pruned_ngrams = [pruned_ngrams[0] if i == 0 else [] for i in range(maxorder)]
    pruned_ngrams[0].append(tuple([num_tokens]))
    blank_grams[tuple([num_tokens])] = True
    for gram in all_grams:
        # Iterate over all possibilities by using a vector of 0s, 1s to 
        # denote whether a blank is being used at each position
        if blank == "optional":
            # given a gram ab.. of order n, we have have n+1 positions
            # avaiable whether to use blank or not.
            onehot_vectors = itertools.product([0, 1], repeat=len(gram) + 1)
        elif blank == "forced":
            # must include a blank token in between
            onehot_vectors = [[1] * (len(gram) + 1)]
        else:
            raise ValueError(
                "Invalid value specificed for blank. Must be in |optional|forced|none|"
            )
        for j in onehot_vectors:
            new_array = []
            for idx, oz in enumerate(j[:-1]):
                if oz == 1 and gram[idx] != START_IDX:
                    new_array.append(num_tokens)
                new_array.append(gram[idx])
            if j[-1] == 1:
                new_array.append(num_tokens)

            for n in range(maxorder):
                for e in range(n, len(new_array)):
                    cur_gram = tuple(new_array[e - n : e + 1])
                    if num_tokens in cur_gram and cur_gram not in blank_grams:
                        pruned_ngrams[n].append(cur_gram)
                        blank_grams[cur_gram] = True
    return pruned_ngrams


def parse_lines(lines, lexicon):
    with open(lexicon, "r") as fid:
        lex = (l.strip().split() for l in fid)
        lex = {l[0]: l[1:] for l in lex}
    return [[t for w in l.split(WORDSEP) for t in lex[w]] for l in lines]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute data stats.")
    parser.add_argument("--data_path", type=str, help="Path to dataset.")
    parser.add_argument(
        "--tokens", type=str, help="Path to token list (in order used with training)."
    )
    parser.add_argument("--lexicon", type=str, help="Path to lexicon.", default=None)
    parser.add_argument(
        "--prune",
        metavar="N",
        type=int,
        nargs="+",
        help="Threshold values to prune unigrams, bigrams, etc.",
    )
    parser.add_argument(
        "--blank",
        default="none",
        choices=['none', 'optional', 'forced'],
        help="Specifies the usage of blank token"
        "'none' - do not use blank token "
        "'optional' - allow an optional blank inbetween tokens"
        "'forced' - force a blank inbetween tokens (also referred to as garbage token)",
    )
    parser.add_argument(
        "--save_path", default=None, help="Path to save transition graph."
    )
    args = parser.parse_args()

    for i, j in zip(args.prune[:-1], args.prune[1:]):
        if i > j:
            raise ValueError("Pruning values must be non-decreasing.")

    print(f"Building {len(args.prune)}-gram transition model")

    # Build table of counts and then back-off if below threshold
    with open(args.data_path, "r") as fid:
        lines = [l.strip() for l in fid]
    with open(args.tokens, "r") as fid:
        tokens = [l.strip() for l in fid]
    if args.lexicon is not None:
        lines = parse_lines(lines, args.lexicon)
    tokens_to_idx = {t: e for e, t in enumerate(tokens)}

    ngram = len(args.prune)
    print("Counting data...")
    ngrams = count_ngrams(lines, ngram, tokens_to_idx)

    pruned_ngrams = prune_ngrams(ngrams, args.prune)
    for n in range(ngram):
        print(f"Kept {len(pruned_ngrams[n])} of {len(ngrams[n])} {n+1}-grams")

    if args.blank != "none":
        pruned_ngrams = add_blank_grams(pruned_ngrams, len(tokens_to_idx), args.blank)

    print("Building graph from pruned ngrams...")
    graph = build_graph(pruned_ngrams)
    print("Graph has {} arcs and {} nodes.".format(graph.num_arcs(), graph.num_nodes()))

    if args.save_path is not None:
        print(f"Saving graph to {args.save_path}")
        with open(args.save_path, "w") as fid:
            graph_str = graph.__repr__().strip()
            fid.write(graph_str)
