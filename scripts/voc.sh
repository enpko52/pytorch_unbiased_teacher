#!/bin/sh

# Create dataset directory
mkdir -p ./datasets/VOC2007 
mkdir -p ./datasets/VOC2012 

# Download the PASCAL VOC datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P ./scripts
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P ./scripts
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P ./scripts

# Decompress the dataset files
tar -xf ./scripts/VOCtrainval_06-Nov-2007.tar -C ./scripts
tar -xf ./scripts/VOCtest_06-Nov-2007.tar -C ./scripts
tar -xf ./scripts/VOCtrainval_11-May-2012.tar -C ./scripts

# Move the dataset to ./data directory
mv ./scripts/VOCdevkit/VOC2007/* ./datasets/VOC2007
mv ./scripts/VOCdevkit/VOC2012/* ./datasets/VOC2012

# Delete the resources
rm -rf ./scripts/VOCdevkit
rm -rf ./scripts/VOCtrainval_06-Nov-2007.tar
rm -rf ./scripts/VOCtest_06-Nov-2007.tar
rm -rf ./scripts/VOCtrainval_11-May-2012.tar
