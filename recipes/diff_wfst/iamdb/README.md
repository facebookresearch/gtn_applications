## Preparing the dataset

Register [here](https://fki.tic.heia-fr.ch/login) and run the following script to download the dataset:
```
sh ../../../datasets/download/iamdb.sh <path_to_data> <email> <password>
```

## Dense Transitions

| ngram      | No blank | Forced blank | Optonal blank |
| ----------- | ----------- | ----------- | ----------- |
| 0      | [ltr_ngram0_noblank.json](ltr_ngram0_noblank.json)  | [ltr_ngram0_forceblank.json](ltr_ngram0_forceblank.json)   | [ltr_ngram0_optblank.json](ltr_ngram0_optblank.json)       |
| 1   | [ltr_ngram1_noblank.json](ltr_ngram1_noblank.json)       | [ltr_ngram1_forceblank.json](ltr_ngram1_forceblank.json) | [ltr_ngram1_optblank.json](ltr_ngram1_optblank.json)       |
| 2   | [ltr_ngram2_noblank.json](ltr_ngram2_noblank.json)        | [ltr_ngram2_forceblank.json](ltr_ngram2_forceblank.json) | [ltr_ngram2_optblank.json](ltr_ngram2_optblank.json)       |


## Pruned Transitions

For preparing the list of letters and 1K wordpieces, use the following the script replacing `<...>` with appropriate path
```
python ../../../datasets/iamdb.py --data_path <path_to_data> --save_text <...>/train.txt --save_tokens <...>/letters.txt

python ../../../scripts/make_wordpieces.py --dataset iamdb --data_dir <path_to_data> --text_file <...>/train.txt --output_prefix <...>/iamdb_1kwp --num_pieces 1000
```

List of letters will be located at `<...>/letters.txt` and token, lexicon set for 1K wordpieces can be found at `<...>/iamdb_1kwp_tokens_1000.txt`, `<...>/iamdb_1kwp_lex_1000.txt`. 

For creating pruned transitions graph (with backoff), use the following the script 
```
# ltr, prune = 0
python ../../../scripts/build_transitions.py --data_path <...>/train.txt --tokens <...>/letters.txt --blank optional --add_self_loops --save_path  <...>/ltr_prune_0_0_optblank.bin --prune 0 0

# ltr, prune = 10
python ../../../scripts/build_transitions.py --data_path <...>/train.txt --tokens <...>/letters.txt --blank optional --add_self_loops --save_path  <...>/ltr_prune_0_0_optblank.bin --prune 0 10

# 1kwp, prune = 0
python ../../../scripts/build_transitions.py --data_path <...>/train.txt --tokens <...>/iamdb_1kwp_tokens_1000.txt --lexicon <...>/iamdb_1kwp_lex_1000.txt --blank optional --add_self_loops --save_path  <...>/1kwp_prune_0_0_optblank.bin --prune 0 0

# 1kwp, prune = 10
python ../../../scripts/build_transitions.py --data_path <...>/train.txt --tokens <...>/iamdb_1kwp_tokens_1000.txt --lexicon <...>/iamdb_1kwp_lex_1000.txt --blank optional --add_self_loops --save_path  <...>/1kwp_prune_0_0_optblank.bin --prune 0 10
```

| Pruning      | Letters | 1K Wordpieces | 
| ----------- | ----------- | ----------- | 
| None      | [ltr_ngram2_optblank_prunenone.json](ltr_ngram2_optblank_prunenone.json)  | [1kwp_ngram2_optblank_prunenone.json](1kwp_ngram2_optblank_prunenone.json)   | 
| 0   | [ltr_ngram2_optblank_prune0.json](ltr_ngram2_optblank_prune0.json)       | [1kwp_ngram2_optblank_prune0.json](1kwp_ngram2_optblank_prune0.json) | 
| 10   | [ltr_ngram2_optblank_prune10.json](ltr_ngram2_optblank_prune10.json)        | [1kwp_ngram2_optblank_prune10.json](1kwp_ngram2_optblank_prune10.json) | 



