## Benchmarks

A list of results for various benchmarks.


### IAM Handwriting Database

The [data and task definition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) are publically available. Note that the current task definition seems to be somewhat different than prior work. Hence, at the moment reported numbers are not directly comparable until we are able to reconcile the differences.

Publication | Test WER | Test CER | Val WER | Val CER | Model Description |
---|---|---|---|---|---|
[Graves et al. 2009](https://www.cs.toronto.edu/~graves/tpami_2009.pdf) | 25.9 | 18.2 | - | - | Bi-directional LSTM with CTC, <br/> Bi-gram LM + Dictionary decoding |
[Kozielski et al. 2013](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.392.8113&rep=rep1&type=pdf) | 13.3 | 5.1 | 9.5 | 2.7 | Hybrid HMM/LSTM, <br/> Tri-gram LM |
[Pham et al. 2014](https://arxiv.org/abs/1312.4569) | 13.6 | 5.1 | 11.2 | 3.7 | MDLSTM + CTC with dropout, <br/> Tri-gram LM
[Pham et al. 2014](https://arxiv.org/abs/1312.4569) | 35.1 | 10.8 | 27.3 | 7.4 | MDLSTM + CTC with dropout, <br/> No LM
