#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ $# -ne 3 ]; then
    echo "Script to download IAM Handwriting Database"
    echo "Usage: $0 datadir email password"
    echo "* datadir: directory to download the data"
    echo "* email, password: Register here - https://fki.tic.heia-fr.ch/login and supply the credentials"
    exit
fi

datadir=$1
email=$2
password=$3


credentials=$username:$password

mkdir -p $datadir
URL=https://fki.tic.heia-fr.ch/DBs/iamDB

echo "\nLogging in and creating a session ..."
curl -X POST --cookie-jar ./cookies.txt --data "email=${email}&password=${password}" https://fki.tic.heia-fr.ch/login

echo "\nDownloading metadata ..."
curl --cookie ./cookies.txt -o $datadir/ascii.tgz $URL/data/ascii.tgz
tar -xzf $datadir/ascii.tgz -C $datadir 

echo "\nDownloading task definition ..."
curl -o $datadir/largeWriterIndependentTextLineRecognitionTask.zip  https://fki.tic.heia-fr.ch/static/zip/largeWriterIndependentTextLineRecognitionTask.zip
unzip $datadir/largeWriterIndependentTextLineRecognitionTask.zip -d $datadir 

echo "\nDownloading images ..."
for form in 'A-D' 'E-H' 'I-Z'
do
  curl --cookie ./cookies.txt -o $datadir/forms${form}.tgz $URL/data/forms${form}.tgz
  tar -xzf $datadir/forms${form}.tgz -C $datadir 
done

echo "\nDone!"
