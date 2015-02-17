import os
import sys
import subprocess
from __builtin__ import len
import random
import cv2
import numpy as np
import theano
import theano.tensor as T
import csv

class LoadPics(object):
    """Routines for loading pictures"""

    def __init__(self, path):
        random.seed(42)
        self.path = path
        # Order of the submission
        # Creating Training -> Training Dev, and Testset
        classes = os.listdir(path)
        try:
            classes.remove('.DS_Store')
        except:
            pass
        d = {}
        l = 0
        self.testsets = {}
        self.trainingsets = {}
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None
        self.numberOfClassed = len(classes)
        self.classes = classes
        for cc in classes:
            imgs = os.listdir(path + cc)
            random.shuffle(imgs) #Shuffels in place in one class
            d[cc] = len(imgs)
            l += len(imgs)

            testStart = int(len(imgs) * 0.75)
            testEnd = (len(imgs)) - 1
            # A small set for debuggnig
            # testStart = int(len(imgs) * 0.05)
            # testEnd = testStart + 5

            self.trainingsets[cc] = imgs[:testStart]
            self.testsets[cc] =  imgs[testStart:testEnd]
        print("Number of classes " + str(len(self.trainingsets)) + " number of images " + str(l))


    def writeTraining(self, training=True, outfile = None):
        fc = csv.reader(file('sampleSubmission.csv.head.csv'))
        fout = open(outfile, 'w');
        head = fc.next()[1:]
        c = 0
        for i, name in enumerate(head):
            if training:
                files = self.trainingsets[name];
            else:
                files = self.testsets[name];
            for f in files:
                s = str(self.path) + name + "/" + str(f) + " " + str(i)
                #print(s)
                fout.write(s + '\n')
                c += 1
        fout.close();
        print("Number of written files " + str(c))




if __name__ == '__main__':
    #path = "/Users/oli/Proj_Large_Data/kaggle_plankton/train_resized/"
    path = "/home/dueo/data_kaggel_bowl/train_resized60x60/"
    d = LoadPics(path)
    d.writeTraining(training=True, outfile='training_60x60.txt')
    d.writeTraining(training=False, outfile='test.txt60x60')
    print("Finished, creating files")


