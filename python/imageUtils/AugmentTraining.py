import csv
import cv2
import os
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage.morphology import black_tophat, white_tophat, disk

__author__ = 'oli'

show = False
#rots = (30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330)


def writeImg(path, img, kind, file, mani_num):
    dir = path + "/" + kind
    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(dir + '/' + file + '_' + str(mani_num) + '.jpg', img)

selem = disk(1)
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 3:
        print "Usage: python AugmentTraining inputFileList outPut"
        exit(1)
    outPath = sys.argv[2]
    filesOrg = []
    reader = csv.reader(open(sys.argv[1]))
    lineNum = 0
    for rowNum,row in enumerate(reader):
        filename = str.split(row[0])[0]
        cc = str.split(filename,'/')
        file = str.split(cc[-1],'.')[0]
        kind = cc[-2]
        if (rowNum % 100 == 0):
          print(str(rowNum) + kind + " " + file)

        img_org = cv2.imread(filename)
        # Hack um es zu zerstoeren
        #img_org = cv2.imread('/home/dueo/data_kaggel_bowl/train_augmented/trichodesmium_puff/44350_0.jpg')
        if show:
            cv2.imshow('ORI', img_org)
        mani_num = 0;
        writeImg(outPath, img_org, kind, file, mani_num) #Original
        rots = np.random.uniform(0,360,10).astype(int) #10 random rotations
        for i, rot in enumerate(rots):
            im_size = img_org.shape[0]
            if (np.random.rand() > 0.5):
                if (np.random.rand() > 0.5):
                    img_org = cv2.flip(img_org,0)
                else:
                    img_org = cv2.flip(img_org,1)
            scale = np.random.uniform(0.7,1.3)
            mat = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), rot, scale=scale)
            img_rotated = cv2.warpAffine(img_org, mat, (im_size, im_size), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

            img_out = np.zeros((img_rotated.shape[0], img_rotated.shape[1], 3), dtype=np.uint8)
            img_orig = img_rotated[:,:,0]
            img_btop = 255-black_tophat(img_orig, selem)
            img_wtop = 255-white_tophat(img_orig, selem)
            img_out[:, :, 1] = img_btop
            img_out[:, :, 2] = img_wtop

            img_rotated = img_out

            if show:
              cv2.imshow('Rot_' + str(i), img_rotated)
            writeImg(outPath, img_rotated, kind, file, i+1) #Original
        if show:
            cv2.waitKey(20000 )
            lineNum += 1
            if (lineNum > 20):
                break


