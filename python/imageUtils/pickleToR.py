import pickle
import numpy as np

if (False):
    (X_train, y_train) = pickle.load(open("/Users/oli/Proj_Large_Data/PiVision/pivision/trunk/python/pickeledStuff/training_48x48_aligned_large.p", "r"))
    print("Loaded " + str(len(y_train)) + " training data")
    import csv
    filename = "/Users/oli/Proj_Large_Data/PiVision/pivision/trunk/python/pickeledStuff/training_48x48_aligned_large.p_R"
    w = csv.writer(open(filename + '.csv', 'w'))
    for c in range(0, len(X_train)):
        d = np.append(y_train[c], X_train[c].reshape(-1))
        w.writerow(d)

(X, y_names, names4Index, y) = pickle.load(open("/Users/oli/Proj_Large_Data/PiVision/pivision/trunk/python/pickeledStuff/testing_48x48_aligned_large.p", "r"))
print("Loaded " + str(len(y)) + " training data")
manual = {'Dejan':0,'Diego':1,'Martin':2,'Oliver':3,'Rebekka':4,'Ruedi':5}
import csv
filename = "/Users/oli/Proj_Large_Data/PiVision/pivision/trunk/python/pickeledStuff/testing_48x48_aligned_large.p_R"
w = csv.writer(open(filename + '.csv', 'w'))
for c in range(0, len(y_names)):
    d = np.append(int(manual[y_names[c]]), X[c].reshape(-1))
    w.writerow(d)
