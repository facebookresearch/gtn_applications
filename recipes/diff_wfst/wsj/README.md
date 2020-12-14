## Preparing the dataset

```
export ROOT=<path_to_repo>
```

Prepare the WSJ dataset:

```
cd $ROOT/datasets
python preprocess_wsj.py --data_path <path_to_wsj> --save_path <path_to_save_jsons> --convert
```

Make the word piece token and lexicons sets:
```
cd $ROOT/scripts
for np in 200 500 1000 1500
do
  python make_wordpieces.py \
    --dataset wsj \
    --data_dir <path_to_save_jsons> \
    --output_prefix <path_to_save_word_pieces>/word_pieces \
    --num_pieces $np
done
```

## Marginalized decompositions

Edit the following JSON files to use the correct path to the word piece token and lexicon files.

| # Tokens | No Marginilization | Marginalized Decompositions |
| ----------- | ----------- | ----------- |
| 200      | [200wp_no_decomps.json](200wp_no_decomps.json)   | [200wp_marginalized_decomps.json](200wp_marginalized_decomps.json)   | 
| 500      | [500wp_no_decomps.json](500wp_no_decomps.json)   | [500wp_marginalized_decomps.json](500wp_marginalized_decomps.json)   | 
| 1000     | [1000wp_no_decomps.json](1000wp_no_decomps.json) | [1000wp_marginalized_decomps.json](1000wp_marginalized_decomps.json) | 
| 1500     | [1500wp_no_decomps.json](1500wp_no_decomps.json) | [1500wp_marginalized_decomps.json](1500wp_marginalized_decomps.json) | 
