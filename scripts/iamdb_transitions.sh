#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. This source code is
# licensed under the MIT license found in the LICENSE file in the root
# directory of this source tree.

DATA=<data_path>
TRAIN_TEXT=/tmp/iamdb_train_text.txt
TRAIN_TOKENS=/tmp/iamdb_train_tokens.txt
WP_TOKENS=/checkpoint/$USER/data/iamdb/word_pieces_tokens_1000.txt
WP_LEXICON=/checkpoint/$USER/data/iamdb/word_pieces_lex_1000.txt

# Pruning values
P1=0
P2=5
P3=10
SAVE_DIR=/checkpoint/$USER/data/iamdb/transitions_wp1k_${P1}_${P2}_${P3}.txt
BLANK="--blank optional"

# Step 1: Save the tokenized training text
python ../datasets/iamdb.py --data_path $DATA --save_text $TRAIN_TEXT --save_tokens $TRAIN_TOKENS 

# Step 2: Build the transition graph
python build_transitions.py --data_path $TRAIN_TEXT --tokens $WP_TOKENS --lexicon $WP_LEXICON --save_path $SAVE_DIR --prune $P1 $P2 $P3 $BLANK
