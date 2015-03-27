'''
    
'''

# See http://stackoverflow.com/questions/7117797/export-matlab-variable-to-text-for-python-usage
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import h5py
import numpy as np



def createTrainingData(file = '/Users/oli/Proj_Large_Data/Deep_Learning_MRI/BRATS-2/brats2.h5', show=True):
    lnum = 0
    with h5py.File(file, 'r') as f:
        for name in f:
            t1c = np.asarray(f[name + '/' + 'VSD.Brain.XX.O.MR_T1c'])
            pred = np.asarray(f[name + '/' + 'VSD.Brain_3more.XX.XX.OT'])
            if show:
                fig = plt.figure()
                plt.title(name)
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(hspace=1e-3, wspace=1e-3)
            for i, z in enumerate(range(65, 145, 5)):
                tc1s = (np.array(t1c[z, 20:180, 0:160], dtype='float32')).reshape(1,1,160,160)
                preds = (np.array(pred[z, 20:180, 0:160], dtype='uint8')).reshape(1,1,160,160)
                if (lnum == 0):
                    X = tc1s
                    Y = preds
                else:
                    X = np.vstack((X, tc1s))
                    Y = np.vstack((Y, preds))
                if show:
                    a = fig.add_subplot(6, 6, (2 * i + 1), xticks=[], yticks=[])  # NB the one based API sucks!
                    plt.imshow(X[lnum,0,:,:], cmap=plt.get_cmap('gray'))
                    a = fig.add_subplot(6, 6, (2 * i + 2), xticks=[], yticks=[])  # NB the one based API sucks!
                    plt.imshow(Y[lnum,0,:,:], cmap=plt.get_cmap('gray'))
                lnum += 1
            if show:
                plt.pause(1)
    return X,Y

if __name__ == '__main__':
    import time
    start = time.time()
    X,Y = createTrainingData(show=False)
    print("Time " + str(time.time() - start))

    import pickle
    with open('data/data.pkl', "wb") as f:
        pickle.dump([X,Y], f)


