__author__ = 'oli'
# Simple tester to be used to check cv2 installation on the Raspberry

import cv2
import numpy as np
import time
from PIL import Image as Image

face_cascade = cv2.CascadeClassifier()
stat = face_cascade.load('models/haarcascade_frontalface_alt.xml')
if (stat == False):
    print('Could not load the face detector')
else:
    print('Loaded the face detector')


def scale_image(image, scale_factor=1.0):
        if image is not None:
            new_size =  ( int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor) )
            return cv2.resize(image, new_size)
        else:
            return None

def getFaces(image, scale_factor = 0.3):
        if image is None:
            return None

        # if it is a color image - convert to grayscale for faster detection
        if len(image.shape)>2 :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale image - this allows a faster face detection - detect faces on small images - result for large image
        scaled_image = scale_image(image, scale_factor)
        faces_scaled = face_cascade.detectMultiScale(scaled_image)

        # faces = self.face_detector.detect(image)
        faces = None
        if faces_scaled is not None and len(faces_scaled)>0:
            faces = faces_scaled / scale_factor
            faces = faces.astype(int)
            pass

        #print 'successfully detected the following faces: ', faces
        return faces



img = cv2.imread("examples/Oliver-2-16.png", 0)
img = np.asarray(img)
t_start_viola = time.time()
face_list = getFaces(img)
time_viola_jones = time.time() - t_start_viola
print(" Time for detection (Viola & Jones) : " +  "%06.2f msec "%(time_viola_jones * 1000))

if face_list is not None and len(face_list)>0:
            print("Processing " + str(face_list))
            t_start = time.time()
            x_0, y_0, w, h = face_list[0]   # assume there is exactly one face
            x_1 = (x_0 + w)
            y_1 = (y_0 + h)
            img_face = img[y_0:y_1, x_0:x_1]
            Size_For_Eye_Detection = (48, 48)
            img_face = cv2.resize(img_face, Size_For_Eye_Detection, Image.ANTIALIAS)
            faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
            mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
            cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0, 360, 255, -1)
            img_face = cv2.bitwise_and(mask, img_face)
