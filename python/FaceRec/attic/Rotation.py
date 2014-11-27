__author__ = 'oli'

import numpy as np

# Auch noch drehungen aus der ebene
    # http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
    # http://stackoverflow.com/questions/6606891/opencv-virtually-camera-rotating-translating-for-birds-eye-view
    w = im_size
    h = im_size
    alpha = 2.1
    beta = 2.1
    gamma = -1.1
    f = 500#Keine Ahnung was der Parameter f soll?

    A1 = np.float32(
       [
           [1, 0, -w/2],
           [0, 1, -h/2],
           [0, 0,    0],
           [0, 0,    1]
       ]
    )

    RX = np.float32(
        [
          [1,          0,           0, 0],
          [0, np.cos(alpha), -np.sin(alpha), 0,],
          [0, np.sin(alpha),  np.cos(alpha), 0,],
          [0,0,0,1]
        ]
    )

    RY = np.float32(
        [
            [np.cos(beta), 0, -np.sin(beta), 0],
            [       0, 1,          0, 0],
            [np.sin(beta), 0,  np.cos(beta), 0],
            [0, 0,          0, 1]
        ]
    )

    RZ = np.float32(
        [
           [np.cos(gamma), -np.sin(gamma), 0, 0],
           [np.sin(gamma),  np.cos(gamma), 0, 0],
           [0,0,1,0],
           [0,0,0,1]
        ])

    R = np.dot(RX, np.dot(RY, RZ))

    A2 = np.float32(
        [
            [f, 0, w/2, 0],
            [f, 0, w/2, 0],
            [0, 0,   1, 0]
        ]
    )

    trafo = np.dot(A2, np.dot(R, A1))
    img_rotated = cv2.warpPerspective(img_rotated, trafo, (im_size, im_size))