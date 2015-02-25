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

    def __init__(self, path, trainingFraction):
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

            testStart = int(len(imgs) * trainingFraction)
            testEnd = (len(imgs)) - 1
            # A small set for debuggnig
            # testStart = int(len(imgs) * 0.05)
            # testEnd = testStart + 5

            self.trainingsets[cc] = imgs[:testStart]
            self.testsets[cc] =  imgs[testStart:testEnd]
        print("Number of classes " + str(len(self.trainingsets)) + " number of images " + str(l))


    def writeTraining(self, training=True, outfile = None, sample='sampleSubmission.csv.head.csv'):
        fc = csv.reader(file(sample))
        fout = open(outfile, 'w');
        head = fc.next()[1:]
        c = 0
        lines = []
        for i, name in enumerate(head):
            if training:
                files = self.trainingsets[name];
            else:
                files = self.testsets[name];
            for f in files:
                s = str(self.path) + name + "/" + str(f) + " " + str(i)
                lines.append(s)
                c += 1
        random.shuffle(lines)
        for line in lines:
            fout.write(line + '\n')
        fout.close();
        print("Number of written files " + str(c))




if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print "Usage: python CreateLists input_folder namesFile prefix trainingFrac"
        exit(1)
    path   = sys.argv[1]
    sample = sys.argv[2]
    trainingFaction = sys.argv[4]

    d = LoadPics(path, float(trainingFaction))
    d.writeTraining(training=True, outfile=sys.argv[3] + 'train.txt', sample=sample)
    d.writeTraining(training=False, outfile=sys.argv[3] + 'test.txt', sample=sample)
    print("Finished, creating files")


