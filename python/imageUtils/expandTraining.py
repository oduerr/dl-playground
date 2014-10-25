__author__ = 'oli'

import numpy as np

import math
import csv
import gzip
import cv2

# Creates "new" training data, by rotating the old pixels
def expandTraining(filename, filenameExp):
    x_tmp = []
    y_tmp = []
    with gzip.open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            vals = np.asarray(row[1:], np.int)
            y_tmp.append(int(row[0]))
            x_tmp.append(np.asarray(row[1:], np.int))
    print("Read Images ")
    cv2.namedWindow('Original Training Set', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Distorted Training Set', cv2.WINDOW_NORMAL)
    c = 0
    rot = range(-8,8,1)
    w = csv.writer(open(filenameExp + '.csv', 'w'))
    for row in x_tmp:
        print(c)
        vals = np.asarray(row)
        y = y_tmp[c]
        d = np.append(y, row)
        w.writerow(d)
        c += 1
        NDumm = int(math.sqrt(len(vals)))
        img = np.reshape(vals, (NDumm, NDumm)) / 255.0
        cv2.imshow('Original Training Set', img)

        for r in rot:
            mat = cv2.getRotationMatrix2D((NDumm/2, NDumm/2), r, 1)
            img_rotated = cv2.warpAffine(img, mat, (NDumm, NDumm))
            cv2.imshow('Distorted Training Set', img_rotated)
            d = np.append(y,  np.asarray(img_rotated.reshape(-1) * 255, dtype=np.int))
            w.writerow(d)
            cv2.waitKey(1)


if __name__ == '__main__':
    filenameTraining = "../../data/training_48x48_unaligned_large.p_R.csv.gz"
    filenameTraining_expanded = "../../data/training_48x48_unaligned_large_expanded.p_R.csv.gz"
    expandTraining(filenameTraining, filenameTraining_expanded)