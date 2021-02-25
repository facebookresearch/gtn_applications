#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ $# -ne 1 ]; then
    echo "Script to download Librispeech"
    echo "Usage: $0 datadir "
    echo "* datadir: directory to download the data"
    exit
fi

datadir=$1

mkdir -p $datadir

for f in train-clean-100.tar.gz dev-other.tar.gz dev-clean.tar.gz test-other.tar.gz test-clean.tar.gz; do 
    wget https://www.openslr.org/resources/12/${f} -O ${datadir}/${f} 
    tar -xzf ${datadir}/${f} -C $datadir
done 
