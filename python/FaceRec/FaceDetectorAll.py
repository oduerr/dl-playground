__author__ = 'oli'
# Taken from the piVision Projekt


import cv2
import PreProcessor
from convolutional_mlp_face import LeNet5State, LeNet5Topology
import LeNetPredictor as LeNetPredictor
import numpy as np
import time as time
import os as os
import Sources
from PIL import Image as Image

class FaceDetectorAll:

    def __init__(self):
        print('Trying to load face detector')
        self.face_cascade = cv2.CascadeClassifier()
        stat = self.face_cascade.load('models/haarcascade_frontalface_alt.xml')
        if (stat == False):
            print('Could not load the face detector')
        else:
            print('Loaded the face detector')
        self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/state.p', deepOut=True)
        self.ok = 0
        self.all = 1e-16
        self.wrong = 0
        print("Loaded the face predictor")

    # simply scale the image by a given factor
    def scale_image(self, image, scale_factor=1.0):
        if image is not None:
            new_size =  ( int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor) )
            return cv2.resize(image, new_size)
        else:
            return None

    def getFaces(self, image, scale_factor = 0.3):
        if image is None:
            return None

        # if it is a color image - convert to grayscale for faster detection
        if len(image.shape)>2 :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale image - this allows a faster face detection - detect faces on small images - result for large image
        scaled_image = self.scale_image(image, scale_factor)
        faces_scaled = self.face_cascade.detectMultiScale(scaled_image)

        # faces = self.face_detector.detect(image)
        faces = None
        if faces_scaled is not None and len(faces_scaled)>0:
            faces = faces_scaled / scale_factor
            faces = faces.astype(int)
            pass

        #print 'successfully detected the following faces: ', faces
        return faces

    def preprocess(self, img_face):
        Size_For_Eye_Detection = (48, 48)
        faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
        img_face = cv2.resize(img_face, Size_For_Eye_Detection, Image.ANTIALIAS)
        # mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
        # cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0, 360, 255, -1)
        # img_face = cv2.bitwise_and(mask, img_face)
        X = np.asarray(img_face)
        X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
            + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
            + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])
        return X, img_face

    def processImage(self, img, y=-1, writer = None):
        img_org = img.copy()
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Grey Scaled
        face_list = fd.getFaces(img)
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(1, figsize=(18, 12))
        if face_list is not None and len(face_list)>0:
            t_start = time.time()
            x_0, y_0, w, h = face_list[0]   # assume there is exactly one face
            x_1 = (x_0 + w)
            y_1 = (y_0 + h)
            img_face = img[y_0:y_1, x_0:x_1]

            X, img_face = self.preprocess(img_face)

            if writer is not None:
                d = np.append(y, X.reshape(-1))
                writer.writerow(d)

            X = X / 255.

            res = self.pred.getPrediction(X)

            #cv2.imshow('Extracted', cv2.resize(img_face, (200,200)))
            #cv2.imshow('LocalBinaryHists', cv2.resize(X, (200,200)))

            pos = np.arange(6)+.5
            names = ('Dejan', 'Diego', 'Martin', 'Oliver', 'Rebekka', 'Ruedi')
            predName   = str(names[int(res.argmax())])
            predPValue = res[0][res.argmax()]
            frame_width = 1
            if y is not None:
                if y == int(res.argmax()):
                    if (predPValue > 0.9):
                        self.ok += 1
                    frame_col = (0,255,0)
                else:
                    frame_col = (128,0,0)
                    if (predPValue > 0.9):
                        frame_col = (255,0,0)
                        self.wrong += 1
                        print("Wrong")
            if (predPValue > 0.9):
                self.all += 1
                frame_width = 4

            print("Time for the whole pipeline " + str(time.time() - t_start) + " overall acc. " + str(float(self.ok) / self.all) + " wrong " + str(self.wrong))

            plt.clf()
            plt.subplot(421)
            plt.title('Original Image')
            cv2.rectangle(img_org,(x_0,y_0),(x_0+h,y_0+h), frame_col,frame_width)
            cv2.putText(img_org,str(predName + " (" + str(round(predPValue,3)) + ")") ,(x_0,y_0+h), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_col, 2)

            plt.subplot(422)
            plt.yticks(pos, names)
            plt.barh(pos, np.asarray(res[0], dtype = float), align='center')
            plt.title("Logistic Regression " + predName + " " + str(round(predPValue,2)) + " acc. " + str(round(1.0 * self.ok / self.all, 2)))
            plt.axes([0, 1,0, 7])

            plt.subplot(421)
            plt.imshow(img_org)
            plt.subplot(423)
            face = plt.imshow(img_face)
            plt.title("Detected Face ")
            face.set_cmap('gray')
            plt.subplot(424)
            dd = plt.imshow(X)
            plt.title("Preprocessed Face ")
            dd.set_cmap('gray')

            # Kernels of Layer 0
            #plt.subplot(425)
            plt.subplot2grid((4,2),(2,0), colspan=2)
            plt.title('Result after first max-pooling layer')
            maxPool0 = self.pred.getPool0Out(X)[0]
            nkerns0 = maxPool0.shape[0]
            s0 = maxPool0.shape[1]
            ddd = plt.imshow(np.reshape(maxPool0, (s0, s0 * nkerns0)))
            ddd.set_cmap('gray')

            # Kernels of Layer 1
            plt.subplot2grid((4,2),(3,0), colspan=2)
            maxPool1 = self.pred.getPool1Out(X)[0]
            nkerns1 = maxPool1.shape[0]
            s1 = maxPool1.shape[1]
            nkerns1 = min(nkerns1, 10)
            plt.title('Result after second max-pooling layer. ' + str(nkerns1) + " of " + str(maxPool1.shape[0]) + " images")
            dddd = plt.imshow(np.reshape(maxPool1[0:nkerns1], (s1, nkerns1 * s1)))
            dddd.set_cmap('gray')

            # f2 = plt.figure(2, figsize=(18, 16))
            # dddd = plt.imshow(np.reshape(maxPool1, (s1, nkerns1 * s1)))
            # dddd.set_cmap('gray')

            plt.draw()

        #cv2.imshow('Original', img_org)
        #cv2.waitKey(100)


if __name__ == "__main__":
    print("Hallo Gallo")
    fd = FaceDetectorAll()

    if (False): #Using the webcam
        from utils import ImageCapturer
        cap = ImageCapturer.ImageCapturer()
        if not cap:
            print "Error opening capture device"
        else:
            print "successfully imported video capture"
        while True:
            rval, frame = cap.get_image()
            fd.processImage(frame)

    if (True):
        img_path = os.path.abspath('/Users/oli/Proj_Large_Data/PiVision/pivision/images/session_30_july_2014')
        [y, block, names, filenames] = Sources.read_images_2(img_path, useBatch=1, maxNum=2000)

        w = None
        import csv
        w = csv.writer(open("../../data/" + 'batch1_46_lph.csv', 'w'))

        for (idx, file_name) in enumerate(filenames):
            img = cv2.imread(file_name)
            fd.processImage(img, y[idx], w)


