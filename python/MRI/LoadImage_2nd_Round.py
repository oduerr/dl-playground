__author__ = 'oli'

import os
import sys
from os import walk
import dicom
import matplotlib.pyplot as plt

base = "/Users/oli/Proj_Large_Data/Deep_Learning_MRI/Sample_21_1_2015/GBM"
patient = "AAYFX7JFVOJJ3KCZBKBFETVJTY======"
date = "20140521-0";

method = "1-localizer";
method = "3-ep2d_diff_M128_b0_1000_DIN_ADC";
method = "6-t2_spc_ns_sag_p2_iso";
method = "7-spctra";
method = "4-t2_tse_tra_512_5mm";
method = "2-ep2d_diff_M128_b0_1000_DIN";
method = "5-t1_fl2d_tra";


baseName = base + "/" + patient + "/" + patient + "-" + date + "/" + method


if __name__ == "__main__":
    print("Hallo Gallo")
    print(baseName)
    files = os.listdir(baseName)
    c = 1
    for file in files:
        print(file)
        try:
            plan = dicom.read_file(baseName + "/" + file)
            plt.subplot(5,5,c)
            pix = plan.pixel_array
            print(plan.InstanceNumber)
            plt.title("In Num " + str(plan.InstanceNumber))
            plt.imshow(pix)
            c += 1
        except:
            print "Unexpected error:", sys.exc_info()[0]

    plt.draw()
    plt.waitforbuttonpress()

