Disclaimer this is just to see how caffee works. The training and testset are much to small. Further we set the training set equal the testset which is dumb.

### Training the model

We need

* ```lenet_solver.prototxt``` for specifying the SGD
* ```lenet_train_test_files.prototxt``` to specify the architecture of the CNN
* And since we have not used a LEVELDB or LMDB we have to provide a list of the files together with labels for training and testing. ```train_files.txt```. The file content should be as follows:
```
images/amphipods/30718.jpg 0
images/amphipods/22289.jpg 0
images/hydromedusae_h15/103353.jpg 1
images/hydromedusae_h15/111275.jpg 1
...
```

Call ```python TestCNN.py``` or do
```
~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt
```

### Validation of the mode
* TODO

### Deploying (make the prediction)
A new prototype file is needed (this will change in the future). You can copy it from ```lenet_train_test_files.prototxt``` but be sure to remove the stuff which needs labels.

