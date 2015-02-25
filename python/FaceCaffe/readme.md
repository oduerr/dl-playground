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
We have two batches of images (batch1 indoor, batch2 taken outdoors). We use batch2 to test the performance and we split batch1 in two parts, one to train the classifier and the other one to evaluate the performance (also called testset in the caffe). Caffe works on data-layes, these can come from a database (see later) or they can come from lists of files (see below). We create the lists of files using the following 2 commands.
```
  ~/dl-playground/python/FaceCaffe$ python ../imageUtils/CreateLists.py /home/dueo/dl-playground/data/images/batch1/ names2Numbers.txt batch1_ 0.80
  ~/dl-playground/python/FaceCaffe$ python ../imageUtils/CreateLists.py /home/dueo/dl-playground/data/images/batch2/ names2Numbers.txt batch2_ 1000.0
```
The last entry (0.8, 10000) determines the fraction of the training-set. So from batch1 80% is for training and 20% for testing. In batch 2 we only have one set, which we will use for validation later on. The first lines of batch1_train.txt are:
```
/home/dueo/dl-playground/data/images/batch1/Martin/107.png 2
/home/dueo/dl-playground/data/images/batch1/Rebekka/230.png 4
/home/dueo/dl-playground/data/images/batch1/Martin/137.png 2
/home/dueo/dl-playground/data/images/batch1/Martin/136.png 2
/home/dueo/dl-playground/data/images/batch1/Rebekka/203.png 4
/home/dueo/dl-playground/data/images/batch1/Oliver/154.png 3
/home/dueo/dl-playground/data/images/batch1/Ruedi/281.png 5
/home/dueo/dl-playground/data/images/batch1/Ruedi/255.png 5
/home/dueo/dl-playground/data/images/batch1/Dejan/40.png 0
```
Note that the script ```CreateLists.py``` does a random shuffling. If we would not do this random shuffeling it could be that only images of on person are in a particular mini-batch of the training set.


## 

