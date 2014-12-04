__author__ = 'oli'

import numpy as np

import math
import csv
import gzip
import cv2

show = True

#rot = (-8,-6,-4,4,6,8)
#rot = (-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) #paper 5
#rot = (-6,-5,-4,-3,3,4,5,6) #paper 8
#rot = (-6,-4,4,6) #paper 13
rot = (-6,-5,-4,-3,3,4,5,6) #paper 19


dists = (-2,-1,1,2)

# Creates "new" training data, by rotating the old pixels
def distorb(img):
    im_size = img.shape[0]
    r = rot[np.random.randint(0, len(rot))]
    #paper 12 0.05 --> 0.1
    #paper 15 0.1 --> 0.05 ?? Not sure in paper 16 it's 0.1
    #paper 16 scale = np.random.uniform(0.9,1.1)
    #paper 18 scale = np.random.uniform(0.95,1.05)
    #paper 20 scale = np.random.uniform(0.95,1.05)
    scale = np.random.uniform(0.9,1.10)
    mat = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), r, scale=scale)
    dist = 0
    if (np.random.uniform() < 0.5):
        dist = dists[np.random.randint(0, len(dists))]
        mat[0, 2] = mat[0, 2] + dist
    if (np.random.uniform() < 0.5):
        dist = dists[np.random.randint(0, len(dists))]
        mat[1, 2] = mat[1, 2] + dist

    # primat[1,2] = mat[1,2] + dist                    nt(dist)
    img_rotated = cv2.warpAffine(img, mat, (im_size, im_size))
    # Add some noise
    img_rotated = np.multiply(img_rotated, np.random.binomial(size = img_rotated.shape, n = 1, p = 1 - 0.2))
    # paper 11 noise from 0.1 to 0.15
    # paper 12 back to 0.1
    # paper 15 back to 0.2

    # Some rolling of the angles to miminc the camera movement
#     rows,cols = img_rotated.shape[:2]
#     #  ---> erste Komponente
#     #  |
#     #  V Zweite
#     srcTri = np.array([(0,0),          (cols-1,0),  (0,rows-1)], np.float32)
#     # Corresponding Destination Points. Remember, both sets are of float32 type
#     rol = np.random.uniform(-0.05,0.05)#rol = nach hinten abkippen 0 kein Abkippen
#     left = np.random.uniform(-0.05,0.05) #um z-achse
#     dstTri = np.array([(0.0, rows*rol), (cols-1, rows*rol), (-left * (cols-1),rows-1)],np.float32)
#     warp_mat = cv2.getAffineTransform(srcTri,dstTri)   # Generating affine transform matrix of size 2x3
    
    #img_rotated = cv2.warpAffine(img_rotated,warp_mat,(cols,rows))


    return img_rotated


def expandTraining(filename, filenameExp):
    x_tmp = []
    y_tmp = []
    y_table = np.zeros(10)
    with gzip.open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            vals = np.asarray(row[1:], np.int)
            y = int(row[0])
            y_tmp.append(y)
            y_table[y] += 1
            x_tmp.append(np.asarray(row[1:], np.int))
    print("Read Images ")
    if (show):
        cv2.namedWindow('Original Training Set', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Distorted Training Set', cv2.WINDOW_NORMAL)
    c = 0
    w = csv.writer(open(filenameExp + '.csv', 'w'))
    print("Writing to " + filenameExp)
    for row in x_tmp:
        print("Manipulating original image %d", c)
        vals = np.asarray(row)
        y = y_tmp[c]
        d = np.append(y, row)
        # w.writerow(d) we do not write the original training set, idea from Cice...
        c += 1
        im_size = int(math.sqrt(len(vals)))
        img = np.reshape(vals, (im_size, im_size)) / 255.0
        if show: cv2.imshow('Original Training Set', img)
        reps = int(10 * y_table.sum() / y_table[y] )
        for i in xrange(reps):
            img_rotated = distorb(img)
            d = np.append(y,  np.asarray(img_rotated.reshape(-1) * 255, dtype=np.int))
            w.writerow(d)
            if show:
                cv2.imshow('Distorted Training Set', cv2.resize(img_rotated,  (img_rotated.shape[0] *  2, img_rotated.shape[1] *  2)))
                cv2.waitKey(1)

if __name__ == '__main__':
    filenameTraining = "../../data/batch1_46_lph.csv.gz"
    filenameTraining_expanded = "../../data/batch1_46_lph_extended_100"
    expandTraining(filenameTraining, filenameTraining_expanded)
    print("Packing")
    import subprocess
    print(subprocess.check_output(['gzip', filenameTraining_expanded + ".csv"]))
