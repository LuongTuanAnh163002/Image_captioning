#!/bin/bash
# Flickr8k dataset http://cocodataset.org
# Download command: bash ./scripts/get_flickr8k.sh

# Download/unzip dataset
d='./' # unzip directory
file_id='1P-32Vfy3-s8gaAxbLqTbjLAWlKDGzbTy' # ID file
url="https://drive.google.com/uc?export=download&id=$file_id"
filename='flickr8k.zip'

gdown $file_id

echo 'Unzipping' $filename '...'
unzip -q $filename -d $d

echo 'Removing' $filename '...'
rm $filename

echo 'Download complete!'
