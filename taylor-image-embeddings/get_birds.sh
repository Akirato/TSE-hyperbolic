#!/usr/bin/env bash
cd data/cub
wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xzvf CUB_200_2011.tgz
rm CUB_200_2011.tgz
rm attributes.txt
mkdir images
mv CUB_200_2011/images/*/*.jpg images
rm -r CUB_200_2011
cd ../..
