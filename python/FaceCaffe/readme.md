# Experimenting with Caffe
A short tutorial how to create a classifier from images.

## Creating the images (prerequisite)
In this repository there images but they are stored in a single, so in the first step we create the directories and convert the images to pngs.

```
dueo@srv-lab-t-706:~/dl-playground/python/imageUtils$ ./mkdirs.sh 
dueo@srv-lab-t-706:~/dl-playground/python/imageUtils$ python CreateImages.py 
```
In the data directory, you should find a subfolder with two batches, of 48x48 images like the one below.
![sample image](imgs/0.png)

## Creating the lists of images
We have two batches of images (batch1 indoor, batch2 taken outdoors). We use batch2 to test the performance and we split batch1 in two parts, one to train the classifier and the other one to evaluate the perfromance (also called testset in the caffe). Caffe works on data-layes, these can come from a database (see later) or they can come from lists of files like this one:
```
images/amphipods/30718.jpg 0
images/amphipods/22289.jpg 0
images/hydromedusae_h15/103353.jpg 1
images/hydromedusae_h15/111275.jpg 1
...
```
We create those files using the following python skript.




