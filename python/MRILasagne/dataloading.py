__author__ = 'oli'
import pickle
import numpy as np
from sklearn.utils import shuffle

class loadSimpleData:

    PIXELS = -1

    def load_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        X,y = data[0]
        for i in range(1,len(data)): #We put all together since the splitting into training and test set is done anyway
            xc, yc = data[i]
            X = np.vstack((X, xc))
            y = np.hstack((y, yc))
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        X, y = shuffle(X, y, random_state=42)
        self.PIXELS = int(np.sqrt(X.shape[1]))
        print ("Shape of X " + str(X.shape) + " Number of pixels " + str(self.PIXELS))
        print (" Min / Max X " + str(np.min(X)) + " " + str(np.max(X)))
        print (" Min / Max Y " + str(np.min(y)) + " " + str(np.max(y)))
        print (" Shape of Y " + str(y.shape))
        return X,y

    def load2d(self, filename):
        X, y = self.load_data(filename) #

        # Batch normalization
        Xmean = X.mean(axis = 0)
        XStd = np.sqrt(X.var(axis=0))
        X = 100*(X-Xmean)/(XStd + 0.01)

        print ("AAAfter Batchnormalization Min / Max X / Mean " + str(np.min(X)) + " / " + str(np.max(X)) + " / " + str(np.mean(X)))
        X = X.reshape(-1, 1, self.PIXELS, self.PIXELS) #shape e.g. (2101, 1, 56, 56)
        return X, y

if __name__ == '__main__':
    sl = loadSimpleData()
    X,y = sl.load2d('data/data56.pkl')
    print("Loaded data")


