## Benchmarks

A list of results for various benchmarks.


### IAM Handwriting Database

The [data and task definition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) are publically available. Note that the current task definition seems to be somewhat different than prior work. Hence, at the moment reported numbers are not directly comparable until we are able to reconcile the differences.

Publication | Test WER | Test CER | Model Description |
---|---|---|---|
[Graves et al. 2009](https://www.cs.toronto.edu/~graves/tpami_2009.pdf) | 25.9 | 18.2 | Bi-directional LSTM with CTC, <br/> Bi-gram LM + Dictionary decoding |
