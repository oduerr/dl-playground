#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

/home/dueo/caffe/caffe/build/tools/compute_image_mean /home/dueo/data_kaggel_bowl/train_lmdb_64 \
  /home/dueo/data_kaggel_bowl/train_mean_64.binaryproto

echo "Done."
