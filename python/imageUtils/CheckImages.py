__author__ = 'oli'

import cv2
import os
import numpy as np

path = "/Users/oli/Proj_Large_Data/kaggle_plankton/train/"
classes = os.listdir(path)
try:
    classes.remove('.DS_Store')
except:
    pass

fout = open('sizes.txt', 'w')
for c in classes:
    files = os.listdir(path + c)
    try:
        files.remove('.DS_Store')
    except:
        pass
    for f in files:
        fN = path + "/" + c + "/" + f
        print (fN)
        img = cv2.imread(fN)
        s = np.shape(img)
        line = c + "\t" + str(s[0]) + "\t" + str(s[1]) + "\t" + f
        fout.write(line + "\n")
        print(line)
fout.close()

