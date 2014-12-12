__author__ = 'oli'

from os import walk
import dicom
import matplotlib.pyplot as plt

files = []
for (dirpath, dirnames, filenames) in walk("/Users/oli/Proj_Large_Data/Deep_Learning_MRI/ep2d_diff_3scan_trace_p3_ADC/"):
    files.extend(filenames)
    break

fig = plt.figure("Analysis of the test-files", figsize=(18, 12))
c = 0
for file in files:
    print(file)
    plan = dicom.read_file(dirpath + file)
    plt.subplot(5,5,c)
    pix = plan.pixel_array
    print(plan.InstanceNumber)
    plt.title("In Num " + str(plan.InstanceNumber))
    plt.imshow(pix)
    c += 1

plt.draw()
plt.waitforbuttonpress()