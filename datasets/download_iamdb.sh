#!/bin/bash
# Register to download the data: http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php
# Supply credentials below:
datadir=$1
username="awnihannun"
password="awniiamdb3"

credentials=$username:$password

mkdir -p $datadir/iamdb && cd $datadir/iamdb
URL=http://www.fki.inf.unibe.ch/DBs/iamDB

# Download meta data
curl -u $credentials -O $URL/data/ascii/ascii.tgz
tar -xzvf ascii.tgz

# Task definition
curl -O $URL/tasks/largeWriterIndependentTextLineRecognitionTask.zip
unzip largeWriterIndependentTextLineRecognitionTask.zip

# Download images
for form in 'A-D' 'E-H' 'I-Z'
do
  curl -u $credentials -O $URL/data/forms/forms${form}.tgz
  tar -xzvf forms${form}.tgz
done
