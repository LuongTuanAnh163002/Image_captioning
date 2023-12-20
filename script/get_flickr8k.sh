#!/bin/bash
# Flickr8k dataset http://cocodataset.org
# Download command: bash ./scripts/get_flickr8k.sh

# Download/unzip dataset
d='./' # unzip directory
gdown 1P-32Vfy3-s8gaAxbLqTbjLAWlKDGzbTy

echo 'Unzipping' $filename '...'
unzip -q flickr8k.zip -d $d

echo 'Removing' $filename '...'
rm flickr8k.zip

echo 'Download complete!'
