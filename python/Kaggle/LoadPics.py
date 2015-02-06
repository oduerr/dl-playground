import os
import sys
import subprocess
from __builtin__ import len
import matplotlib.pyplot as plt
import random


class LoadPics(object):
    """Routines for loading pictures"""

    def __init__(self, path):
        self.path = path
        # Creating Training -> Training Dev, and Testset
        classes = os.listdir(path)
        d = {}
        l = 0
        self.testsets = {}
        self.trainingsets = {}
        for cc in classes:
            imgs = os.listdir(path + cc)
            random.shuffle(imgs) #Shuffels in place
            d[cc] = len(imgs)
            l += len(imgs)
            sp = int(len(imgs) * 0.75)
            self.trainingsets[cc] = imgs[:sp]
            self.testsets[cc] = imgs[sp:]
        print("Number of classes " + str(len(self.trainingsets)) + " number of images " + str(l))





if __name__ == '__main__':
    path = "/Users/oli/Proj_Large_Data/kaggle_plankton/train"
    LoadPics(path)
