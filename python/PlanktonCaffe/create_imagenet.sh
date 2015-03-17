#!/usr/bin/env sh
# Taken from ~/caffe/caffe/examples/imagenet/create_imagenet.sh


echo "Creating train lmdb..."

GLOG_logtostderr=1 /home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=256 \
    --resize_width=256 \
    --shuffle \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/train_full.txt \
    /home/dueo/data_kaggel_bowl/train_lmdb_256

echo "Creating val lmdb..."

GLOG_logtostderr=1 /home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=256 \
    --resize_width=256 \
    --shuffle \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/test_full.txt \
    /home/dueo/data_kaggel_bowl/val_lmdb_256

GLOG_logtostderr=1 /home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --gray \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/test_kaggle_full.txt \
    /home/dueo/data_kaggel_bowl/val_kaggle_lmdb_128

GLOG_logtostderr=1 /home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=256 \
    --resize_width=256 \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/full_augmented_train.txt \
    /home/dueo/data_kaggel_bowl/train_augmented_lmdb_256




echo "Done..."

# EXAMPLE=examples/imagenet
# # Dort wird reingeschrieben
# DATA=data/ilsvrc12 
# TOOLS=/home/dueo/caffe/caffe/build/tools/
# 
# # 
# TRAIN_DATA_ROOT=/home/dueo/data_kaggel_bowl/imagenet/train/
# VAL_DATA_ROOT=/home/dueo/data_kaggel_bowl/imagenet/val/
# 
# # Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# # already been resized using another tool.
# RESIZE=false
# if $RESIZE; then
#   RESIZE_HEIGHT=256
#   RESIZE_WIDTH=256
# else
#   RESIZE_HEIGHT=0
#   RESIZE_WIDTH=0
# fi
# 
# if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#   echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#   echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#        "where the ImageNet training data is stored."
#   exit 1
# fi
# 
# if [ ! -d "$VAL_DATA_ROOT" ]; then
#   echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#   echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#        "where the ImageNet validation data is stored."
#   exit 1
# fi
# 
# echo "Creating train lmdb..."
# 
# GLOG_logtostderr=1 $TOOLS/convert_imageset \
#     --resize_height=$RESIZE_HEIGHT \
#     --resize_width=$RESIZE_WIDTH \
#     --shuffle \
#     $TRAIN_DATA_ROOT \
#     $DATA/train.txt \
#     $EXAMPLE/ilsvrc12_train_lmdb
# 
# echo "Creating val lmdb..."
# 
# GLOG_logtostderr=1 $TOOLS/convert_imageset \
#     --resize_height=$RESIZE_HEIGHT \
#     --resize_width=$RESIZE_WIDTH \
#     --shuffle \
#     $VAL_DATA_ROOT \
#     $DATA/val.txt \
#     $EXAMPLE/ilsvrc12_val_lmdb
# 
# echo "Done."
