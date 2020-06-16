## Benchmarks

A list of results for various benchmarks.


### IAM Handwriting Database

The [data and task definition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) are publically available. Note that the current task definition seems to be somewhat different than prior work. Hence, at the moment reported numbers are not directly comparable until we are able to reconcile the differences.

Publication | Test WER | Test CER | Val WER | Val CER | Model Description |
---|---|---|---|---|---|
[Graves et al. 2009](https://www.cs.toronto.edu/~graves/tpami_2009.pdf) | 25.9 | 18.2 | - | - | Bi-directional LSTM with CTC, <br/> Bigram LM + Dictionary decoding |
[Kozielski et al. 2013](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.392.8113&rep=rep1&type=pdf) | 13.3 | 5.1 | 9.5 | 2.7 | Hybrid HMM/LSTM, <br/> Trigram LM |
[Pham et al. 2014](https://arxiv.org/abs/1312.4569) | 13.6 | 5.1 | 11.2 | 3.7 | MDLSTM + CTC with dropout, <br/> Trigram LM
[Pham et al. 2014](https://arxiv.org/abs/1312.4569) | 35.1 | 10.8 | 27.3 | 7.4 | MDLSTM + CTC with dropout, <br/> No LM
[Voigtlaender et al. 2016](https://www.vision.rwth-aachen.de/media/papers/MDLSTM_final.pdf) | 9.3 | 3.5 | 7.1 | 2.4 | Multi-dimensional LSTM + CTC,<br/> Trigram LM, paragraph level eval
[Bluche et al. 2017](http://www.tbluche.com/files/icdar17_gnn.pdf) | 10.5 | 3.2 | - | - | Gated Convolutional RNN + CTC,<br/> Trigram LM,<br/>+ 130k lines data
[Kang et al. 2020](https://arxiv.org/abs/2005.13044) | 24.5 | 7.6 | - | - | Transformer, <br/> No LM
[Kang et al. 2020](https://arxiv.org/abs/2005.13044) | 15.5 | 4.7 | - | - | Transformer + synthetic data, <br/> No LM 
