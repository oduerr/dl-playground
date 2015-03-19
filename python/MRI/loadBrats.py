import os
import sys
import mha

file1 = '/home/dueo/data/BRATS/BRATS-2/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.685.mha'
img=mha.new(input_file=file1, data_type='long')
img.read_mha(file1)

#from medpy.io import load
#image_data, image_header = load(file1)
