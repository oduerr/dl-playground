import numpy as np
import cv2
import platform
if platform.machine() == 'armv6l':  # for the raspberry
    WEBCAMPI = True
    import picam
else:
    WEBCAMPI = False

class ImageCapturer():
    def __init__(self):
        if WEBCAMPI:
            print("---- Using the picam ----")
        else:
            print("---- Using the webcam ----")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 320 )
            self.cap.set(4, 240 )
            # if self.cap.isOpened() ??
        pass

    def __del__(self):
        if WEBCAMPI:
            # some picam release?
            pass
        else:
            self.cap.release()
            print("camera released!!")

    def get_image(self):
        frame = None
        rval = False
        if WEBCAMPI:
            frametemp = picam.takePhotoWithDetails(640, 480, 100)   # width, height, quality
            frame = np.array(frametemp)
            rval = frame != None # check frame.size ??
        else:
            rval, frame = self.cap.read()

        return rval, frame