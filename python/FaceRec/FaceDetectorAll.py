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
import Utils_dueo
import csv

# Parameters
scale_fac = 0.3
borderProb = 0.85
show = True
webcam = False
rocWriter = csv.writer(open('roc.csv', 'w'))



class FaceDetectorAll:

    def __init__(self, show = False):
        self.show = show
        self.faces = 0.0
        print('Trying to load face detector')
        self.face_cascade = cv2.CascadeClassifier()
        stat = self.face_cascade.load('models/haarcascade_frontalface_alt.xml')
        if (stat == False):
            print('Could not load the face detector')
        else:
            print('Loaded the face detector')
        #self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/state_lbh_elip_K100_batch3', deepOut=True)
        #self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/state_lbh_elip_K100_batch3_long_training', deepOut=True)
        #self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/good_ones/state_lbh_elip_K100_batch3___Hat__Nur__2__Error_wenn_Ueber_90Prozent', deepOut=True)
        #self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/good_ones/k20.p', deepOut=True)
        self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/good_ones/state_lbh_elip_scale_K100', deepOut=True)

        #self.pred = LeNetPredictor.LeNetPredictor(stateIn='models/good_ones/k100_lr0.1_speckel.p', deepOut=True)
        self.ok = 0
        self.all = 1e-16
        self.wrong = 0
        print("Loaded the face predictor")

    # simply scale the image by a given factor
    def scale_image(self, image, scale_factor=scale_fac):
        if image is not None:
            new_size =  ( int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor) )
            return cv2.resize(image, new_size)
        else:
            return None

    def getFaces(self, image, scale_factor = scale_fac):
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
        img_face = cv2.resize(img_face, Size_For_Eye_Detection, Image.ANTIALIAS)
        img_norm = Utils_dueo.LBH_Norm(img_face)
        img_norm = Utils_dueo.mask_on_rect(img_norm)
        return img_norm, img_face



    # def preprocess2(self, img_face):
    #     Size_For_Eye_Detection = (46,46)
    #     img_face = cv2.resize(img_face, Size_For_Eye_Detection, Image.ANTIALIAS)
    #     img_face = PreProcessor.PreProcessor.gamma_filter(img_face, gamma=0.5)
    #     img_face = PreProcessor.PreProcessor.DoG_filter(img_face, sigma_0 = 0.4, sigma_1 = 3.0)
    #     faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
    #     mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
    #     cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0, 360, 255, -1)
    #     img_face = cv2.bitwise_and(mask, img_face)
    #     return img_face, img_face



    def processImage(self, img, y=None, writer = None):

        img_org = img.copy()
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Grey Scaled
        t_start_viola = time.time()
        face_list = fd.getFaces(img)
        time_viola_jones = time.time() - t_start_viola
        if self.show:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure("Hello Convolutional World", figsize=(18, 12))
        if face_list is not None and len(face_list)>0:
            self.faces += 1
            t_start = time.time()
            x_0, y_0, w, h = face_list[0]   # assume there is exactly one face
            x_1 = (x_0 + w)
            y_1 = (y_0 + h)
            img_face = img[y_0:y_1, x_0:x_1]

            X, img_face = self.preprocess(img_face)
            #X, img_face = self.preprocess2(img_face)
            #X = self.mask(X)

            if writer is not None:
                d = np.append(y, X.reshape(-1))
                writer.writerow(d)

            X = X / 255.

            #res = self.pred.getPrediction(X)
            res = self.pred.getPrediction(np.array(X, dtype=np.float32))
            time_cnn = time.time() - t_start

            #cv2.imshow('Extracted', cv2.resize(img_face, (200,200)))
            #cv2.imshow('LocalBinaryHists', cv2.resize(X, (200,200)))

            pos = np.arange(6)+.5
            names = ('Dejan', 'Diego', 'Martin', 'Oliver', 'Rebekka', 'Ruedi')
            predY = int(res.argmax())
            predName   = str(names[predY])
            predPValue = res[0][predY]
            frame_width = 1
            wrong = False
            frame_col = (128,128,128)
            if y is not None:
                if y == predY:
                    if (predPValue > borderProb):
                        self.ok += 1
                    frame_col = (0,255,0)
                else:
                    frame_col = (128,0,0)
                    if (predPValue > borderProb):
                        frame_col = (255,0,0)
                        self.wrong += 1
                        print("Wrong")
                        wrong = True
            #rocWriter.writerow(str(y) + "\t" + str(predY) + "\t" + str(predPValue))
            rocWriter.writerow((y, predY, predPValue))
            if (predPValue > borderProb):
                self.all += 1
                frame_width = 6
            if y is None:
                name = "Unknown"
            else:
                name = names[y]

            print(str(predName) + " (predicted): Time for detection (Viola & Jones) : " +  "%06.2f msec "%(time_viola_jones * 1000)  +
                  " Time for the classific. & prepros. (CNN) : " + "%06.2f msec "%(time_cnn * 1000) + " overall acc. " + str(float(self.ok) / self.all) + " wrong " + str(self.wrong))

            if self.show:
                plt.clf()
                fig.canvas.set_window_title('Test')

                ############## Stats
                fig.text(0.02, 1.00, "Hello Convolutional Network" , fontsize=14, verticalalignment='top')
                fig.text(0.02, 0.97, "Found : " +  str(int(self.all)) + " Right : " + str(self.ok) + " Wrong : " + str(self.wrong) + " Acc. : " + str(round(1.0 * self.ok / self.all, 2))
                         , fontsize=18, verticalalignment='top')
                fig.text(0.02, 0.94, "Time for detection (Viola & Jones)    : " +  "%06.2f"%(time_viola_jones * 1000) + " msec" ,fontsize=12, verticalalignment='top')
                fig.text(0.02, 0.925, "Time for classific. & prepros. (CNN)  : " +  "%06.2f"%(time_cnn * 1000) + " msec" ,fontsize=12, verticalalignment='top')

                ############## Original Image with Box drawn
                plt.subplot(421)
                plt.title('Original Image : ' + str(img_org.shape))
                cv2.rectangle(img_org,(x_0,y_0),(x_0+h,y_0+h), frame_col,frame_width)
                cv2.putText(img_org,str(predName + " (" + str(round(predPValue,3)) + ")") ,(x_0,y_0+h), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_col, 2)
                plt.imshow(img_org)

                ############## Logistic Regression
                plt.subplot(422)
                plt.yticks(pos, names)
                plt.barh(pos, np.asarray(res[0], dtype = float), align='center')
                plt.title("Final Layer (Multinomial Regression) " + predName + " " + str(round(predPValue,2)))

                plt.subplots_adjust(hspace = 0.3)
                ############## Faces
                plt.subplot(423)
                face = plt.imshow(img_face)
                plt.title("Detected Face " + str(img_face.shape))
                face.set_cmap('gray')

                plt.subplot(424)
                dd = plt.imshow(X)
                plt.title("Preprocessed Face " + str(X.shape))
                dd.set_cmap('gray')

                # Kernels of Layer 0
                #plt.subplot(425)
                plt.subplot2grid((4,2),(2,0), colspan=2)
                d = self.pred.getPool0Out(X)
                plt.title('Result after first max-pooling layer ' + str(d.shape))
                maxPool0 = d[0]
                nkerns0 = maxPool0.shape[0]
                s0 = maxPool0.shape[1]
                ddd = plt.imshow(np.reshape(maxPool0, (s0, s0 * nkerns0)))
                ddd.set_cmap('gray')

                # Kernels of Layer 1
                plt.subplot2grid((4,2),(3,0), colspan=2)
                d = self.pred.getPool1Out(X)
                maxPool1 = d[0]
                nkerns1 = maxPool1.shape[0]
                s1 = maxPool1.shape[1]
                nkerns1 = min(nkerns1, 10)
                plt.title('Result after second max-pooling layer. ' + str(d.shape))
                dddd = plt.imshow(np.reshape(maxPool1[0:nkerns1], (s1, nkerns1 * s1)))
                dddd.set_cmap('gray')

                plt.draw()
        print("Cassified " + str(self.all) + " All " + " Acc " + str(round(1.0 * self.ok / self.all, 2)) + " Faces " + str(self.faces))
        #cv2.imshow('Original', img_org)
        #cv2.waitKey(1000000)


if __name__ == "__main__":
    print("Hallo Gallo")
    fd = FaceDetectorAll(show = show)
    if (webcam): #Using the webcam
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
        [y, block, names, filenames] = Sources.read_images_2(img_path, useBatch=2, maxNum=5000)

        w = None
        #import csv
        #w = csv.writer(open("../../data/" + 'batch2_46_gamma_dog.csv', 'w'))

        for (idx, file_name) in enumerate(filenames):
            img = cv2.imread(file_name)
            print("\n Checking Filename " + str(file_name) + " y " + str(y[idx]) )
            fd.processImage(img, y[idx], w)
        print(len(filenames))


