# Training
~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt


### Runing on GPU 1
```
nohup ~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt --gpu=1 > log_lenet.txt&
```