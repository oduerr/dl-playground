import csv
import cv2
import os
import numpy as np

__author__ = 'oli'

show = True
rots = (90,180,270)


def writeImg(path, img, kind, file, mani_num):
    dir = path + "/" + kind
    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(dir + '/' + file + '_' + str(mani_num) + '.jpg', img)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 5:
        print "Usage: python AugmentTraining inputFileList outPut"
        exit(1)
    outPath = sys.argv[2]
    filesOrg = []
    reader = csv.reader(open(sys.argv[1]))
    lineNum = 0
    for row in reader:
        filename = str.split(row[0])[0]
        cc = str.split(filename,'/')
        file = str.split(cc[-1],'.')[0]
        kind = cc[-2]
        print(kind + " " + file)

        img_org = cv2.imread(filename)
        if show:
            cv2.imshow('ORI', img_org)
        mani_num = 0;
        writeImg(outPath, img_org, kind, file, mani_num) #Original

        for i, rot in enumerate(rots):
            im_size = img_org.shape[0]
            scale = np.random.uniform(0.9,1.10)
            mat = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), rot, scale=scale)
            img_rotated = cv2.warpAffine(img_org, mat, (im_size, im_size), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
            cv2.imshow('Rot', img_rotated)
            writeImg(outPath, img_rotated, kind, file, i+1) #Original

        if show:
            cv2.waitKey(200)
        lineNum += 1
        print(row)
        if (lineNum > 20):
            break


