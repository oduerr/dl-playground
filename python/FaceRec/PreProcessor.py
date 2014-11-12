from PIL import Image as Image
import numpy as np
import cv2
#from scipy import ndimage

import FaceDetectorAll

SIZE_FOR_PCA = (100, 100)  # TODO rename
Size_For_Eye_Detection = (100, 100)
#SIZE_FOR_PCA = (48, 48) # in progress for feature extractor
SCALE_FACTOR_EYES = 1.02


class PreProcessor():
    def __init__(self, headless=False):
        self.load_face_detector()
        self.headless = headless
        print 'preprocessor object created'

    def load_face_detector(self):
        self.face_detector = FaceDetectorAll.FaceDetector()

    # a simple conversion from color to grayscale
    def convert_to_grayscale(self, image):
        if len(image.shape) > 2 :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # simply scale the image by a given factor
    def scale_image(self, image, scale_factor=1.0):
        if image is not None:
            new_size =  ( int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor) )
            return cv2.resize(image, new_size)
        else:
            return None


    # ******************* a simple routine to extract all faces of an image ********************************************
    def extract_faces_simple(self, image, scale_factor = 1.0):

        if image is None:
            return None

        # if it is a color image - convert to grayscale for faster detection
        if len(image.shape)>2 :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale image - this allows a faster face detection - detect faces on small images - result for large image
        scaled_image = self.scale_image(image, scale_factor)
        faces_scaled = self.face_detector.detect(scaled_image)

        # faces = self.face_detector.detect(image)
        faces = None
        if faces_scaled is not None and len(faces_scaled)>0:
            print 'hola'
            faces = faces_scaled / scale_factor
            faces = faces.astype(int)
            pass

        print 'successfully detected the following faces: ', faces

        return faces

    # *************************************** gamma filtering **********************************************************
    @staticmethod
    def gamma_filter(face, gamma=1.0):
        if face==None:
            return None
        else:
            # convert to gray scale if it is a color image
            if len(face.shape)>2 :
                # print len(X.shape)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # do gamma filtering
            face = np.array(face, dtype=np.uint8)
            face = np.power(face/255.0, gamma) * 255.0    # gamma correction
            face = np.array(face, dtype=np.uint8)
            return face

    # ************************************ apply Difference of Gaussian (DoG) filtering ********************************
    @staticmethod
    # def DoG_filter(face, sigma_0 = 1.0, sigma_1 = 5.0):
    #     if face==None:
    #         return None
    #     else:
    #         # convert to gray scale if it is a color image
    #         if len(face.shape)>2 :
    #             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #
    #         # Difference of Gaussian filtering
    #         face = np.array(face, dtype=np.int)     # important: convert to int - otherwise funny differences!
    #         face = np.asarray(ndimage.gaussian_filter(face, sigma_0)) - np.asarray(ndimage.gaussian_filter(face, sigma_1))
    #
    #         # shift and rescale so that min value is zero and max value is 255
    #         face = face - np.amin(face)          # start with a minimum of zero
    #         face = face * 255.0 / np.amax(face)  # rescale gray values
    #
    #         # convert back to unsigned int
    #         face = np.array(face, dtype=np.uint8)
    #         return face

    # *************************************** simplified algorithm to align the faces **************************
    def find_eyes(self, face, verbose = False):
        # face_detector = FaceDetector.FaceDetector()
        w = face.shape[0]
        h = face.shape[1]
        # cv2.imshow('face', face)
        # cv2.waitKey(0)
        # split the image vertically into a left and right image - search for one eye in each image
        right_image = face[:, 0:w/2]
        left_image = face[:, w/2:]

        # set parameters for eye detector
        scale_factor_eyes = 1.02
        neighbors = 5
        min_size = (20,20)
        max_size = (45,45)
        # find left eye
        eyes_left =self.face_detector.eye_cascade_left.detectMultiScale(left_image)
        # eyes_left =self.face_detector.eye_cascade_left.detectMultiScale(left_image, scaleFactor=scale_factor_eyes, minNeighbors=neighbors,minSize=min_size, maxSize=max_size)
        if verbose:
            print 'left:', eyes_left


        # find right eye
        eyes_right = self.face_detector.eye_cascade_right.detectMultiScale(right_image, scale_factor_eyes)
        # eyes_right = self.face_detector.eye_cascade_right.detectMultiScale(right_image, scaleFactor=scale_factor_eyes, minNeighbors=neighbors,minSize=min_size, maxSize=max_size)
        if verbose:
            print 'right: ', eyes_right

        # return an image - either None or the image with the two eyes detected
        if (eyes_left is not None and eyes_right is not None and len(eyes_right) == 1 & len(eyes_left) == 1):
            # print("Detected exactly two eyes")
            eyes_left[0,0] += (w/2 - 1)
            return eyes_left, eyes_right
        else:
            return None,None

    # *************************************** simplified algorithm to align the faces **************************
    def find_eyes_fast(self, face, verbose = False):
        print 'fast eye detection'
        # face_detector = FaceDetector.FaceDetector()
        w = face.shape[0]
        h = face.shape[1]

        # split the image vertically into a left and right image - search for one eye in each image
        top = 0
        # bottom = int(h * 2.0/3.0)
        bottom = int(h / 2.0)
        right_image = face[top:bottom, 0:w/2]
        left_image = face[top:bottom, w/2:]

        # set parameters for eye detector
        scale_factor_eyes = 1.02
        neighbors = 5
        min_size = (20,20)
        max_size = (45,45)
        # find left eye
        # eyes_left =self.face_detector.eye_cascade_left.detectMultiScale(left_image)
        eyes_left =self.face_detector.eye_cascade_left.detectMultiScale(left_image, scaleFactor=scale_factor_eyes, minNeighbors=neighbors,minSize=min_size, maxSize=max_size)
        if verbose:
            print 'left:', eyes_left


        # find right eye
        # eyes_right = self.face_detector.eye_cascade_right.detectMultiScale(right_image, scale_factor_eyes)
        eyes_right = self.face_detector.eye_cascade_right.detectMultiScale(right_image, scaleFactor=scale_factor_eyes, minNeighbors=neighbors,minSize=min_size, maxSize=max_size)
        if verbose:
            print 'right: ', eyes_right

        # return an image - either None or the image with the two eyes detected
        if (eyes_left is not None and eyes_right is not None and len(eyes_right) == 1 & len(eyes_left) == 1):
            # print("Detected exactly two eyes")
            eyes_left[0,0] += (w/2 - 1)
            eyes_right[0,1] += top
            eyes_left[0,1] +=top
            return eyes_left, eyes_right
        else:
            return None,None


    # ********************************** a routine that rotates the given image exactly the way we want it ****************
    def align_image(self, face):

        eyes_left, eyes_right = self.find_eyes(face)
        # eyes_left, eyes_right = self.find_eyes_fast(face)

        # return an image - either None or the image with the two eyes detected
        if (eyes_left is not None and eyes_right is not None and len(eyes_right) == 1 & len(eyes_left) == 1):
            # compute angle of rotation
            lx = int( (2*eyes_left[0,0] + eyes_left[0,2])/2 )
            rx = int( (2*eyes_right[0,0] + eyes_right[0,2])/2)
            ly = int(eyes_left[0][1] + eyes_left[0][3] * 0.5)
            ry = int(eyes_right[0][1] + eyes_right[0][3] * 0.5)
            dy = (ly - ry)
            dx = (lx - rx)
            dist = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx) * 180.0 / np.pi
            # print "angle: ", angle

            DESIRED_LEFT_EYE_X = 0.16  #Controls how much of the face is visible after preprocessing.
            DESIRED_LEFT_EYE_Y = 0.20  #0.14
            DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X)
            desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * Size_For_Eye_Detection[0]
            scale = 1.0 * desiredLen / dist  #0.65 von Hand
            scale *=0.9

            center = ((lx + rx) / 2, (ly + ry) / 2)
            mat = cv2.getRotationMatrix2D(center, angle, scale)
            mat[0][2] += Size_For_Eye_Detection[0] * 0.5 - center[0]
            mat[1][2] += Size_For_Eye_Detection[1] * DESIRED_LEFT_EYE_Y - center[1]

            wrapped = cv2.warpAffine(face, mat, Size_For_Eye_Detection)

            # Masking out an elipse
            faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
            if len(face.shape)==2:
                mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
                cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.65)), 0, 0, 360, 255, -1)
            else:
                mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1], 3), np.uint8)
                cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.65)), 0, 0, 360, (255, 255, 255), -1)

            # wrapped = cv2.bilateralFilter(wrapped, 0, 20.0, 2.0)  #From 0.66176 to 0.6764 for Fisher Faces

            # wrapped = cv2.bitwise_and(mask, wrapped)
            return wrapped
        else:
            print("Alignment failed - most probably because of an eye detection problem")
            return None

    # *************** a class that combines various preprocessor methods *******************************************
    def preprocess_color_face(self, color_face, align = True):
        if color_face is not None:
            gray_face = self.convert_to_grayscale(color_face)
            gray_face = cv2.resize(gray_face, Size_For_Eye_Detection, interpolation=Image.ANTIALIAS)
            if align:
                 gray_face = self.align_image(gray_face)
            gray_face = self.gamma_filter(gray_face, gamma=0.5)
            gray_face = self.DoG_filter(gray_face, sigma_0 = 0.4, sigma_1 = 3.0)
            return gray_face
        else:
            return None

    @staticmethod
    def extract_color_face(color_image, face_list, face_index):
        if face_list is not None and len(face_list) > 0:
                    x_0, y_0, w, h = face_list[face_index]   # assume there is exactly one face
                    x_1 = (x_0 + w)
                    y_1 = (y_0 + h)
                    return color_image.get_numpy_array()[y_0:y_1, x_0:x_1]
        else:
            return None

#
# def preProcessDetectedFaces(X):
#     X = cv2.resize(X, SIZE_FOR_PCA, interpolation=Image.ANTIALIAS)
#
#     #TODO: different implementations for preprocessing
#     if (3 > 2):
#         X1 = cv2.equalizeHist(X[0:, 0:SIZE_FOR_PCA[1] / 3])
#         X2 = cv2.equalizeHist(X[0:, SIZE_FOR_PCA[1] / 3:(SIZE_FOR_PCA[1] * 2 / 3)])
#         X3 = cv2.equalizeHist(X[0:, (SIZE_FOR_PCA[1] * 2 / 3):SIZE_FOR_PCA[1]])
#         X = np.append(np.append(X1, X2, 1), X3, 1)
#
#     if (1 > 2):
#         X = cv2.equalizeHist(X)
#
#     if (1 > 2):
#         X = np.asarray(X)
#         X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
#             + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
#             + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
#             + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
#             + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
#             + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
#             + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
#             + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])
#     if (1 > 2):
#         # Simple preprocessing
#
#         # From https://github.com/bytefish/facerec/blob/master/py/facerec/preprocessing.py
#         alpha = 0.1
#         tau = 10.0
#         gamma = 0.2
#         sigma0 = 1.0
#         sigma1 = 2.0
#
#         sigma0 = 2.0
#         sigma1 = 5.0
#
#         X = np.array(X, dtype=np.float32)
#         X = np.power(X, gamma)
#         X = np.asarray(ndimage.gaussian_filter(X, sigma1) - ndimage.gaussian_filter(X, sigma0))
#         X = X / np.power(np.mean(np.power(np.abs(X), alpha)), 1.0 / alpha)
#         X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), tau), alpha)), 1.0 / alpha)
#         X = tau * np.tanh(X / tau)
#
#     return (X)


def DummDummalignFace(face, imgIn, face_detector, headless=False):
    x, y, h, w = face  # It seems that x and y coordinates are mixed
    tox = (x + w)
    toy = (y + h)
    faceCol = imgIn[y:toy, x:tox]
    # ##########
    # Achtung es scheinen 2 Koordinatensystem zu existieren.

    # # openCV
    #  -------> X (erste Koordinate)
    # |
    # |
    # v
    # Y (zweite Koordinate)
    cv2.rectangle(imgIn, (20, 5), (100, 20), (255, 0, 255), 1)

    ## python (wenn man mit X[i,j] ein Bild ausschneidet, dann wie Matrix
    #  -------> j  (erste Koordinate)
    # |
    # |
    # v
    # i (erste Koordinate)
    # Tester:
    # dumm = imgIn[100:120, 50:70]
    # cv2.rectangle(dumm,(0,0),(10,10),(255,0,255),1)

    oL = h / 10  #Overlay
    facecolR = faceCol[0:h / 2 + oL, 0:w / 2 + oL]
    facecolL = faceCol[0:h / 2 + oL,
               w / 2 - oL:]  #Taken from the camera perspective see http://yushiqi.cn/research/eyedetection
    #cv2.waitKey(0)
    eyes_left = face_detector.eye_cascade_right.detectMultiScale(facecolL, SCALE_FACTOR_EYES)

    if not headless:
        cv2.namedWindow('Alignedx2', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Alignedx2', 0, 231)

    for (ex, ey, ew, eh) in eyes_left:
        cv2.rectangle(facecolL, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        cv2.line(facecolL, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        cv2.line(facecolL, (ex + ew, ey), (ex, ey + eh), (0, 255, 0), 1)
        lx = int((ex + ex + ew) / 2 + w / 2 - oL)
    eyes_right = face_detector.eye_cascade_right.detectMultiScale(facecolR, SCALE_FACTOR_EYES)
    for (ex, ey, ew, eh) in eyes_right:
        cv2.rectangle(facecolR, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 1)
        cv2.line(facecolR, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 1)
        cv2.line(facecolR, (ex + ew, ey), (ex, ey + eh), (255, 0, 255), 1)
        rx = int((ex + ex + ew) / 2)  #Relative to the face
        print("Right Eye " + str(eyes_right[0]))
    if (len(eyes_right) == 1 & len(eyes_left) == 1):
        print("Detected exactly two eyes")
        ly = int(eyes_left[0][1] + eyes_left[0][3] * 0.5)
        #cv2.circle(faceCol, (lx,ly), 10, (255,0,0))
        ry = int(eyes_right[0][1] + eyes_right[0][3] * 0.5)
        #cv2.rectangle(img, (rx,ry), (rx+5,ry+5), 255,4)
        #For the transformation code see: https://github.com/MasteringOpenCV/code/blob/master/Chapter8_FaceRecognition/preprocessFace.cpp
        dy = (ly - ry)
        dx = (lx - rx)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        #print("---- " + str(angle))
        DESIRED_LEFT_EYE_X = 0.16  #Controls how much of the face is visible after preprocessing.
        DESIRED_LEFT_EYE_Y = 0.20  #0.14
        DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X)
        desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * SIZE_FOR_PCA[0]
        scale = 0.65 * desiredLen / dist  #0.65 von Hand
        center = ((lx + rx) / 2, (ly + ry) / 2)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0][2] += SIZE_FOR_PCA[0] * 0.5 - center[0]
        mat[1][2] += SIZE_FOR_PCA[1] * DESIRED_LEFT_EYE_Y - center[1]

        #################
        # Masking out an elipse
        faceCenter = (int(SIZE_FOR_PCA[0] * 0.5), int(SIZE_FOR_PCA[1] * 0.4))
        mask = np.zeros((SIZE_FOR_PCA[0], SIZE_FOR_PCA[1], 3), np.uint8)
        #cv2.ellipse(mask, faceCenter, (int(SIZE_FOR_PCA[0] * 0.25), int(SIZE_FOR_PCA[1] * 0.5)),255,-1)
        #cv2.ellipse(mask,(64,64),(12,12),0,0,180,255,-1)
        #cv2.ellipse(mask, faceCenter, (SIZE_FOR_PCA[0] * 0.25, SIZE_FOR_PCA[1] * 0.5), 0,0,180,255,-1)
        #cv2.ellipse(mask, (64,64),(16,64),0,0,360,255,-1)
        cv2.ellipse(mask, faceCenter, (int(SIZE_FOR_PCA[0] * 0.5), int(SIZE_FOR_PCA[1] * 0.65)), 0, 0, 360,
                    (255, 255, 255), -1)
        wrapped = cv2.warpAffine(faceCol, mat, SIZE_FOR_PCA)
        wrapped = cv2.bilateralFilter(wrapped, 0, 20.0, 2.0)  #From 0.66176 to 0.6764 for Fisher Faces
        wrapped = cv2.bitwise_and(mask, wrapped)

        if not headless:
            wrappedShow = cv2.resize(wrapped, (wrapped.shape[1] * 2, wrapped.shape[1] * 2))
            cv2.imshow('Alignedx2', wrappedShow)
            cv2.waitKey(1)
    else:
        print("Not exactly two eyes found")
        return None
    return wrapped

def imshow(title, image, headless):
    if not headless:
        cv2.imshow(title, image)
        cv2.waitKey(1)
