__author__ = 'oli'
import struct
import numpy as np


def readAnotations(path, fileName):
    global f, numByte, fileNameRead, byte, width, decoded, height, all, slices
    f = open(path + fileName, "rb")
    try:
        numByte = 0
        fileNameRead = ""
        byte = f.read(1)
        while byte != "":
            byte = f.read(1)
            if numByte < len(fileName):
                fileNameRead += byte
            else:
                fileNameRead += byte
                width = decoded = struct.unpack('>I', f.read(4))[0]  # This is magic, we have big-endian coding
                height = decoded = struct.unpack('>I', f.read(4))[0]
                #print("Read header of [" + fileNameRead + "] width = , " + str(width) + " height=" + str(height))
                all = np.fromfile(f, dtype='>I', count=-1, sep='')
                slices = len(all) / (width * height)
                Y = np.reshape(all, (slices, width, height))
                #print("Read the rest. We have " + str(slices))
                f.close()
                return Y
            numByte += 1
    finally:
        f.close()

if __name__ == '__main__':
    fileName = '5G4ESHBE4NO7SHGOLRXQTMUDNI======_ep2d_diff_3scan_p3_m128_ADC_3_4.dcm.iov'
    path = "/Users/oli/Proj_Large_Data/Deep_Learning_MRI/insel_annotated/ADC-contouring-test-case/3-ep2d_diff_3scan_p3_m128_ADC/iov/"

    import os
    files = os.listdir(path)
    for file in files:
        Y = readAnotations(path, fileName)
        print(file + ' ' + str(Y.shape) + " " + str(Y.max()))