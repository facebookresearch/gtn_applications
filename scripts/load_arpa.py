import math
import gtn

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


def read_counts_from_arpa(arpa_file):
    fid = open(arpa_file, "r")
    # read header
    while fid.readline().strip() != "\\data\\":
        continue
    line = fid.readline()
    assert "ngram 1" in line
    num_words = int(line.strip().split("=")[1])
    lm_order = 1
    while True:
        line = fid.readline().strip()
        if len(line) == 0:
            break
        lm_order += 1
        assert f"ngram {lm_order}" in line

    counts = []
    vocab = {}
    # read higher order ngrams
    for cur_order in range(1, lm_order + 1):
        counts.append({})
        while f"\\{cur_order}-grams" not in fid.readline():
            continue
        idx = 0
        while True:
            line = fid.readline().strip().split()
            if len(line) == 0 or "\\end\\" == line[0]:
                break
            if cur_order == 1:
                vocab[line[1]] = idx
            gram = line[1 : cur_order + 1]
            key = tuple([vocab[g] for g in gram])
            prob = float(line[0])
            if len(line) > cur_order + 1:
                bckoff = float(line[cur_order + 1])
            else:
                bckoff = 0.0 if cur_order < lm_order else None
            counts[cur_order - 1][key] = (prob, bckoff)
            idx += 1
    assert len(vocab) == num_words
    return counts, vocab


def build_lm_graph(ngram_counts, vocab):
    graph = gtn.Graph(False)
    lm_order = len(ngram_counts)
    assert lm_order > 1, "build_lm_graph doesn't work for unigram LMs"
    state_to_node = {}

    def get_node(state):
        node = state_to_node.get(state, None)
        if node is not None:
            return node
        is_start = state == tuple([vocab[BOS]])
        is_end = vocab[EOS] in state
        node = graph.add_node(is_start, is_end)
        state_to_node[state] = node
        return node

    for counts in ngram_counts:
        for ngram in counts.keys():
            istate, ostate = ngram[0:-1], ngram[len(ngram) - lm_order + 1 :]
            inode = get_node(istate)
            onode = get_node(ostate)
            prob, bckoff = counts[ngram]
            # p(gram[-1] | gram[:-1])
            lbl = ngram[-1] if ngram[-1] != vocab[EOS] else gtn.epsilon
            graph.add_arc(inode, onode, lbl, lbl, prob)
            if bckoff is not None and vocab[EOS] not in ngram:
                bnode = get_node(ngram[1:])
                graph.add_arc(onode, bnode, gtn.epsilon, gtn.epsilon, bckoff)

    return graph


def build_setence_graph(sentence, vocab):
    graph = gtn.Graph(False)
    sidx = [vocab[w] if w in vocab else vocab[UNK] for w in sentence.split()]
    prev = graph.add_node(True, False)
    for e, idx in enumerate(sidx):
        cur = graph.add_node(False, e == len(sidx) - 1)
        graph.add_arc(prev, cur, idx)
        prev = cur
    return graph


if __name__ == "__main__":
    import kenlm
    import os
    import random

    # bigram LM

    sent = "wood pittsburgh cindy jean"
    m = kenlm.Model("lm_small.arpa")
    counts, vocab = read_counts_from_arpa("lm_small.arpa")
    # print(vocab, counts)
    symb = {v: k for k, v in vocab.items()}
    g_lm = build_lm_graph(counts, vocab)
    gtn.write_dot(g_lm, "/tmp/g_lm.dot", symb, symb)
    g_sent = build_setence_graph(sent, vocab)
    gtn.write_dot(g_sent, "/tmp/g_sent.dot", symb, symb)
    g_score = gtn.intersect(g_lm, g_sent)
    gtn.write_dot(g_score, "/tmp/g_score.dot", symb, symb)
    # compare P(sent </s> | <s>)
    assert gtn.viterbi_score(g_score).item() == m.score(sent, bos=True, eos=True)

    # trigram LM
    lm_file = "/tmp/3-gram.pruned.3e-7.arpa"
    if not os.path.exists(lm_file):
        url = "http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz"
        os.system(f"wget {url} -P /tmp/ && gunzip {lm_file}")
        assert os.path.exists(lm_file)

    # lower case arpa file
    with open(lm_file) as f:
        lines = f.readlines()
        with open(lm_file, "w") as f2:
            f2.write("".join([l.lower() for l in lines]))
    m = kenlm.Model(lm_file)
    counts, vocab = read_counts_from_arpa(lm_file)
    symb = {v: k for k, v in vocab.items()}
    g_lm = build_lm_graph(counts, vocab)
    for _ in range(25):
        length = random.randint(1, 20)
        words = [random.choice(list(vocab.keys())) for _ in range(length)]
        sentence = " ".join(words)
        g_sent = build_setence_graph(sentence, vocab)
        g_score = gtn.intersect(g_lm, g_sent)
        kenlm_score = m.score(sentence, bos=True, eos=True)
        gtn_score = gtn.viterbi_score(g_score).item()
        print(f'"{sentence}"; gtn:{gtn_score}; kenlm:{kenlm_score}')
        assert abs(gtn_score - kenlm_score) < 1e-4
