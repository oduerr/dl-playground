__author__ = 'oli'
import os
from __builtin__ import len
import random
import cv2

IMSHOW = False

class LoadPics(object):
    """Routines for loading pictures"""

    def __init__(self, path):
        self.path = path
        # Creating Training -> Training Dev, and Testset
        d = {}
        l = 0
        self.testsets = {}
        self.trainingsets = {}
        print("Number of classes " + str(len(self.trainingsets)) + " number of images " + str(l))


def resize_training():
    path = "/Users/oli/Proj_Large_Data/kaggle_plankton/train/"
    path_new = "/Users/oli/Proj_Large_Data/kaggle_plankton/train_resized/"
    classes = os.listdir(path)
    c = 0
    for cc in classes:
        try:
            os.mkdir(path_new + cc)
        except:
            pass
        imgs = os.listdir(path + cc)
        for img in imgs:
            fin = path + cc + '/' + img
            fout = path_new + cc + '/' + img
            print (fin + " --> " + fout)
            pics = cv2.imread(fin)
            pics_48 = cv2.resize(pics, (46, 46))
            if IMSHOW:
                cv2.imshow('Org', pics)
                cv2.imshow('Resized', pics_48)
                cv2.waitKey(1)
            cv2.imwrite(fout, pics_48)
            c += 1
            print(c)


if __name__ == '__main__':
    resize_training()
    path = "/Users/oli/Proj_Large_Data/kaggle_plankton/test/"
    path_new = "/Users/oli/Proj_Large_Data/kaggle_plankton/test_resized/"
    files = os.listdir(path)
    c = 0
    for fin in files:
        fout = path_new + '/' + fin
        print (fin + " --> " + fout)
        pics = cv2.imread(path + fin)
        pics_48 = cv2.resize(pics, (46, 46))
        if IMSHOW:
            cv2.imshow('Org', pics)
            cv2.imshow('Resized', pics_48)
            cv2.waitKey(1)
        cv2.imwrite(fout, pics_48)
        c += 1
        print(c)




