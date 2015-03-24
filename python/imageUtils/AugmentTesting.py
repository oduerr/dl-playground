import csv
import cv2
import os
import numpy as np

__author__ = 'oli'

show = True
#rots = (30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330)


def writeImg(path, img, kind, file, mani_num):
    dir = path + "/" + kind
    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(dir + '/' + file + '_' + str(mani_num) + '.jpg', img)


def getImage(i, imOriginal):
    if i == 0: #return a central crop of 99x99 pixels
        resized = cv2.resize(imOriginal, (128, 128))
        return resized[14:113, 14:113]
    rot = np.random.uniform(0,360,1).astype(int)[0] #Random rotations
    #rot = 90 * np.random.uniform(0,4,1).astype(int)[0] #Random rotations
#    rot = 0
    im_size = imOriginal.shape[0]
    if (np.random.rand() > 0.5):
       if (np.random.rand() > 0.5):
         img_org = cv2.flip(imOriginal,0)
       else:
         img_org = cv2.flip(imOriginal,1)
    scale = np.random.uniform(0.9,1.1)
    mat = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), rot, scale=scale)
    resized = cv2.warpAffine(imOriginal, mat, (im_size, im_size) , borderValue=(255,255,255))
    resized = cv2.resize(resized, (128, 128))
    #resized = resized * 256.0 - mean #Geht richtig in den Keller auf Werte um 4
#    offsetX = np.random.uniform(10,18,1).astype(int)[0]
#    offsetY = np.random.uniform(10,18,1).astype(int)[0]
    offsetX = np.random.uniform(0,28,1).astype(int)[0] #Random rotations
    offsetY = np.random.uniform(0,28,1).astype(int)[0] #Random rotations
    return resized[offsetY:(99+offsetY), offsetX:(99+offsetX)]




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
        for i in range(0,3):
            img = getImage(i, img_org)
            print(str(np.shape(img)) + " " + str(np.shape(img_org)))
            if show:
              cv2.imshow('Rot_' + str(i), img)
        if show:
            cv2.waitKey(20000 )
            lineNum += 1
            if (lineNum > 20):
                break


