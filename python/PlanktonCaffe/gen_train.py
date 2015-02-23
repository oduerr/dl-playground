import os
import sys
import subprocess

if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -resize 60x60\! "
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(fo + cls)
    except:
        pass
    print(fi + cls)
    imgs = os.listdir(fi + cls)
    for img in imgs:
        md = ""
        md += cmd
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img
        os.system(md)
