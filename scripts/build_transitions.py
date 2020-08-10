import collections

import gtn

START_IDX = -1


def build_graph(ngrams):
    graph = gtn.Graph(False)
    ngram = len(ngrams)
    state_to_node = {}
    start_state = tuple([START_IDX] * (ngram - 1))

    def get_node(state):
        node = state_to_node.get(state, None)
        if node is not None:
            return node
        start = (state == start_state)
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
            istate, ostate = gram[0:-1], gram[len(gram) - ngram + 1:]
            inode = get_node(istate)
            if gram[1:] not in state_to_node:
                raise ValueError(
                    "Ill formed counts: if (x, y_1, ..., y_{n-1}) is above"
                    "the n-gram threshold, then (y_1, ..., y_{n-1}) must be"
                    "above the (n-1)-gram threshold")
            onode = get_node(ostate)
            # p(gram[-1] | gram[:-1])
            graph.add_arc(inode, onode, gram[-1])
    return graph


def count_ngrams(lines, ngram, tokens_to_idx):
    counts = [collections.Counter() for _ in range(ngram)]
    for line in lines:
        # prepend implicit start token
        line = [START_IDX] * (ngram - 1) + [tokens_to_idx[t] for t in line]
        for e in range(ngram - 1, len(line)):
            for n, counter in enumerate(counts):
                counter[tuple(line[e - n:e + 1])] += 1
    return counts


def prune_ngrams(ngrams, prune):
    pruned_ngrams = []
    for n, grams in enumerate(ngrams):
        grams = grams.most_common()
        pruned_grams = [gram for gram, c in grams if c > prune[n]]
        pruned_ngrams.append(pruned_grams)
    return pruned_ngrams


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute data stats.")
    parser.add_argument("--data_path", type=str,
        help="Path to dataset.")
    parser.add_argument("--tokens", type=str,
        help="Path to token list (in order used with training).")
    parser.add_argument("--prune", metavar='N', type=int, nargs='+',
        help="Threshold values to prune unigrams, bigrams, etc.")
    parser.add_argument("--blank", default=False, action="store_true",
        help="Add a blank token between every token in the text data.")
    parser.add_argument("--save_path", default=None,
        help="Path to save transition graph.")
    args = parser.parse_args()

    for i, j in zip(args.prune[:-1], args.prune[1:]):
        if i > j:
            raise ValueError("Pruning values must be non-decreasing.")

    print(f"Building {len(args.prune)}-gram transition model")

    # Build table of counts and then back-off if below threshold
    with open(args.data_path, 'r') as fid:
        lines = [l.strip() for l in fid]
    with open(args.tokens, 'r') as fid:
        tokens = [l.strip() for l in fid]
    tokens_to_idx = {t: e for e, t in enumerate(tokens)}

    ngram = len(args.prune)
    print("Counting data...")
    ngrams = count_ngrams(lines, ngram, tokens_to_idx)

    pruned_ngrams = prune_ngrams(ngrams, args.prune)
    for n in range(ngram):
        print(f"Kept {len(pruned_ngrams[n])} of {len(ngrams[n])} {n+1}-grams")

    print("Building graph from pruned ngrams...")
    graph = build_graph(pruned_ngrams)
    print("Graph has {} arcs and {} nodes".format(
        graph.num_arcs(), graph.num_nodes()))

    if args.save_path is not None:
        with open(args.save_path, 'w') as fid:
            graph_str = graph.__repr__().strip()
            fid.write(graph_str)
