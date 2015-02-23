# Creating images size (256x256) needed for image net
```
  ./create_imagenet.sh
  ./make_imagenet_mean.sh
```

# Starting with
```
  nohup ~/caffe/caffe/build/tools/caffe train -solver solver.prototxt --gpu=1 > imagenet_21_feb.log&
```