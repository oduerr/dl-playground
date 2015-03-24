__author__ = 'oli'
import pickle
import numpy as np
from sklearn.utils import shuffle

class loadSimpleData:

    PIXELS = -1

    def load_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        x,y = data[0]
        for i in range(1,len(data)): #We put all together since the splitting into training and test set is done anyway
            xc, yc = data[i]
            x = np.vstack((x, xc))
            y = np.hstack((y, yc))
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        x, y = shuffle(x, y, random_state=42)
        self.PIXELS = int(np.sqrt(x.shape[1]))
        print ("Shape of X " + str(x.shape) + " Number of pixels " + str(self.PIXELS))
        print ("Shape of Y " + str(y.shape))
        return x,y

    def load2d(self, filename):
        X, y = self.load_data(filename) #

        # Batch normalization
        Xmean = X.mean(axis = 0)
        XStd = np.sqrt(X.var(axis=0))
        X = (X-Xmean)/(XStd + 0.1)

        X = X.reshape(-1, 1, self.PIXELS, self.PIXELS) #shape e.g. (2101, 1, 56, 56)
        return X, y

if __name__ == '__main__':
    sl = loadSimpleData()
    X,y = sl.load2d('data/data56.pkl')
    print("Loaded data")


