__author__ = 'oli'

import cv2
import gzip
import numpy as np
import csv
import math

filenameTraining    = "../../data/training_48x48_unaligned_large.p_R.csv.gz"
filenameValidation = "../../data/testing_48x48_unaligned_large.p_R.csv.gz"
names_for_y =  ('Dejan', 'Diego', 'Martin', 'Oliver', 'Rebekka', 'Ruedi')
show = False

def Mask(vals):
    sizeIn = int(math.sqrt(len(vals)))
    img_face = np.reshape(vals, (sizeIn, sizeIn))
    img_face = mask_on_rect(img_face)
    return np.reshape(img_face, len(vals))

def mask_on_rect(img_face):
    Size_For_Eye_Detection = img_face.shape
    faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
    mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
    cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0,
                360, 255, -1)
    img_face = np.multiply(mask/255, img_face)
    return img_face


def imgsFromCSV(filename, directory):
        y_table = np.zeros(10)
        y_tmp = []
        x_tmp = []
        if (show):
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
        minV = 1e100
        maxV = -1e100
        num = -1
        with gzip.open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                y = int(row[0])
                name = names_for_y[y]
                num += 1
                y_tmp.append(y)
                y_table[y] += 1
                vals = np.asarray(row[1:], np.uint8)
                preprocessed = Mask(vals)
                minV = min(min, np.amin(preprocessed))
                maxV = max(max, np.amax(preprocessed))
                x_tmp.append(preprocessed / 255.)
                n = int(np.sqrt(preprocessed.shape[0]))
                im = np.array(np.reshape(preprocessed,(n,n)), dtype = np.uint8)
                if (show):
                    cv2.imshow('Original', np.reshape(vals/255., (n, n)))
                    cv2.imshow('Preprocessed', im)
                cv2.imwrite(directory + '/' + name + '/' + str(num) + ".png", im)


        print("  Data Range" + str(minV) + "  " + str(maxV))
        print("  Balance " + str(y_table))


if __name__ == '__main__':
    imgsFromCSV(filenameTraining, "../../data/images/batch1")
    imgsFromCSV(filenameValidation, "../../data/images/batch2")



