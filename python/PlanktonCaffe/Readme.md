### Creating the lists with the training and testset
```
python ../imageUtils/CreateLists.py TODO
```
### Creating the databases
```
/home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --shuffle \
    --gray \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/train_full.txt \
    /home/dueo/data_kaggel_bowl/train_lmdb_128
#
/home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --shuffle \
    --gray \
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/test_full.txt \
    /home/dueo/data_kaggel_bowl/test_lmdb_128
```


### Augmenting the training set
```
python ../imageUtils/AugmentTraining.py train_full.txt /home/dueo/data_kaggel_bowl/train_augmented/
```

### Creating lists for the augmentet training set
They are allready the training set so we used 1000 that all belong to the new training set.
```
python ../imageUtils/CreateLists.py /home/dueo/data_kaggel_bowl/train_augmented/ sampleSubmission.csv.head.csv full_augmented_ 1000
```

### Craete the db
```
GLOG_logtostderr=1 /home/dueo/caffe/caffe/build/tools/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --gray\
    / \
    /home/dueo/dl-playground/python/PlanktonCaffe/full_augmented_30_train.txt \
    /home/dueo/data_kaggel_bowl/train_augmented_30_lmdb_128
```

### Calculate mean
```
 ~/caffe/caffe/build/tools/compute_image_mean /home/dueo/data_kaggel_bowl/train_augmented_lmdb_256/ /home/dueo/data_kaggel_bowl/train_augmented_mean.binaryproto
```


### Runing on GPU 1
```
nohup ~/caffe/caffe/build/tools/caffe train -solver solver.prototxt --gpu=1 > log.txt&

# Resuming from previous state
~/caffe/caffe/build/tools/caffe train -solver solver.auto.prototxt --gpu=1 --snapshot models/alexnet_128_test_iter_2900.solverstate

# Starting the whole thig
python -u ../Controller.py > bigrun.txt&

```


### Creating an leveldb data base and using the averages
```
./build/tools/convert_imageset / /home/dueo/dl-playground/python/PlanktonCaffe/training_60x60.txt /home/dueo/data_kaggel_bowl/train_60_lmdb
./build/tools/compute_image_mean /home/dueo/data_kaggel_bowl/train_60_lmdb /home/dueo/data_kaggel_bowl/train_60_mean
/build/tools/convert_imageset -shuffle=true / /home/dueo/dl-playground/python/PlanktonCaffe/test.txt60x60 /home/dueo/data_kaggel_bowl/test_60_lmdb
./build/tools/compute_image_mean /home/dueo/data_kaggel_bowl/test_60_lmdb /home/dueo/data_kaggel_bowl/test_60_mean
```

```

```

### Observations
* Use shuffeling (otherwise one sees oszilations in the data)
* Croping and Mirroring good for training but if deploying on unscaled score approx 2.0