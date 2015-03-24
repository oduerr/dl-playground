__author__ = 'oli'

import csv

fc1 = csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_big6_15000.txt'))
fc2 = csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_big6_21000.txt'))
fc3 = csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_big6_36000.txt'))
fc4 = csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_big8_102k.txt'))
fc5= csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_big8_102k_test_10_aug.txt'))

fout = open('/Users/oli/Proj_Large_Data/kaggle_plankton/submission_6_8aug__join.txt', 'w')
w = csv.writer(fout)

fst =  fc1.next();fc2.next();fc3.next();fc4.next();fc5.next();
w.writerow(fst)


for row1 in fc1:
    row2 = fc2.next()
    row3 = fc3.next()
    row4 = fc4.next()
    row5 = fc5.next()
    newLine = [row1[0]]
    for c in range(1, len(row1)):
        newLine.append(str((float(row1[c]) + float(row2[c]) + float(row3[c]) + float(row4[c]) + float(row5[c]))/5.0))
    w.writerow(newLine);
fout.close()






