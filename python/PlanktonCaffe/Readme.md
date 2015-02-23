# Training
~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt


### Runing on GPU 1
```
nohup ~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt --gpu=1 > log_lenet.txt&
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