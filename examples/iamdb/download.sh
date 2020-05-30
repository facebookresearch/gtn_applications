#!/bin/bash
# Register to download the data: http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php
# Supply credentials below:
username="awnihannun"
password="awniiamdb3"


credentials=$username:$password

mkdir -p data && cd data
URL=http://www.fki.inf.unibe.ch/DBs/iamDB

# Download meta data
#curl -u $credentials -O $URL/data/ascii/ascii.tgz
#tar -xzvf ascii.tgz

# Task definition
#curl -O $URL/tasks/largeWriterIndependentTextLineRecognitionTask.zip
#unzip largeWriterIndependentTextLineRecognitionTask.zip

# Download images
for form in 'A-D' 'E-H' 'I-Z'
do
  curl -u $credentials -O $URL/data/forms/forms${form}.tgz
  tar -xzvf forms${form}.tgz
done
