"""
    Loading as pre-trained LeNet5 for prediction
"""
__author__ = 'oliver'

from convolutional_mlp_plankton import LeNet5State, LeNet5Topology
import LoadPics
from LogisticRegression import LogisticRegression
from LeNetConvPoolLayer import LeNetConvPoolLayer
from HiddenLayer import HiddenLayer

import numpy as np
import theano
from theano import tensor as T
import pickle

import math
import csv
import gzip
import cv2
import csv

class LeNetPredictor(object):

    def __init__(self, stateIn, deepOut = False):
        global pickle

        print("  Loading previous state ...")
        if stateIn.endswith('gz'):
            f = gzip.open(stateIn,'rb')
        else:
            f = open(stateIn, 'r')
        state = pickle.load(f)
        convValues = state.convValues
        w0 = convValues[0][0]
        b0 = convValues[0][1]
        w1 = convValues[1][0]
        b1 = convValues[1][1]
        hiddenVals = state.hiddenValues
        wHidden = hiddenVals[0]
        bHidden = hiddenVals[1]
        logRegValues = state.logRegValues
        wLogReg = logRegValues[0]
        bLogReg = logRegValues[1]
        topo = state.topoplogy
        nkerns = topo.nkerns
        n_out = topo.numLogisticOutput
        print("  Some Values ...")
        print("     Number of Kernels : " + str(nkerns))
        print("     First Kernel w0[0][0] :\n" + str(w0[0][0]))
        print("     bHidden :\n" + str(bHidden))
        print("     bLogReg :\n" + str(bLogReg))
        print("  Building the theano model")
        batch_size = 1

        x = T.matrix('x')   # the data is presented as rasterized images
        layer0_input = x.reshape((batch_size, 1, topo.ishape[0], topo.ishape[1]))
        rng = np.random.RandomState(23455)

        layer0 = LeNetConvPoolLayer(None, input=layer0_input,
                                image_shape=(batch_size, 1, topo.ishape[0],  topo.ishape[0]),
                                filter_shape=(nkerns[0], 1, topo.filter_1, topo.filter_1),
                                poolsize=(topo.pool_1, topo.pool_1), wOld=w0, bOld=b0, deepOut=deepOut)


        layer1 = LeNetConvPoolLayer(None, input=layer0.output,
                                    image_shape=(batch_size, nkerns[0], topo.in_2, topo.in_2),
                                    filter_shape=(nkerns[1], nkerns[0], topo.filter_2, topo.filter_2),
                                    poolsize=(topo.pool_2, topo.pool_2), wOld=w1, bOld=b1, deepOut=deepOut)

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(None, input=layer2_input, n_in=nkerns[1] * topo.hidden_input,
                             n_out=topo.numLogisticInput, activation=T.tanh, Wold = wHidden, bOld = bHidden)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=topo.numLogisticInput, n_out=n_out, Wold = wLogReg, bOld=bLogReg )

        # create a function to compute the mistakes that are made by the model
        # index = T.lscalar()
        # test_model = theano.function([index], layer3.getProbs(),
        #                              givens={x: test_set_x[index * batch_size: (index + 1) * batch_size]})

        self.predict_model = theano.function([x], layer3.getProbs())

        if (deepOut):
            self.layer0_out = theano.function([x], layer0.output)
            self.layer0_conv= theano.function([x], layer0.conv_out)
            self.layer1_conv= theano.function([x], layer1.conv_out)
            self.layer1_out = theano.function([x], layer1.output)
            self.b0 = b0
            self.b1 = b1
            self.w0 = w0
            self.w1 = w1


    def getPrediction(self, imgAsRow):
        """
            :param imgAsRow: integers in the range [0,255]
            :return:
        """
        values=np.reshape(imgAsRow, (46, 46))
        return (self.predict_model(values))

    def getPool0Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer0_out(values))

    def getConv0Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer0_conv(values))

    def getConv1Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer1_conv(values))

    def getPool1Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer1_out(values))


if __name__ == "__main__":
    classesasreadonserver = ['pteropod_butterfly', 'copepod_calanoid_octomoms', 'echinoderm_larva_seastar_brachiolaria', 'detritus_blob', 'fish_larvae_deep_body', 'hydromedusae_bell_and_tentacles', 'copepod_other', 'chaetognath_other', 'hydromedusae_shapeA', 'tornaria_acorn_worm_larvae', 'copepod_cyclopoid_copilia', 'fish_larvae_very_thin_body', 'invertebrate_larvae_other_A', 'trichodesmium_multiple', 'hydromedusae_sideview_big', 'hydromedusae_typeE', 'hydromedusae_liriope', 'copepod_calanoid_small_longantennae', 'euphausiids', 'siphonophore_calycophoran_rocketship_adult', 'appendicularian_straight', 'tunicate_partial', 'pteropod_triangle', 'fecal_pellet', 'protist_noctiluca', 'hydromedusae_typeF', 'detritus_filamentous', 'ephyra', 'fish_larvae_leptocephali', 'copepod_calanoid_eggs', 'hydromedusae_solmundella', 'unknown_unclassified', 'tunicate_doliolid_nurse', 'hydromedusae_haliscera_small_sideview', 'chaetognath_sagitta', 'protist_other', 'echinopluteus', 'acantharia_protist_halo', 'jellies_tentacles', 'trichodesmium_tuft', 'echinoderm_larva_pluteus_brittlestar', 'fish_larvae_myctophids', 'appendicularian_fritillaridae', 'ctenophore_lobate', 'shrimp_zoea', 'echinoderm_larva_seastar_bipinnaria', 'hydromedusae_shapeA_sideview_small', 'unknown_blobs_and_smudges', 'copepod_calanoid_eucalanus', 'protist_fuzzy_olive', 'tunicate_salp_chains', 'trichodesmium_bowtie', 'siphonophore_calycophoran_sphaeronectes_young', 'appendicularian_s_shape', 'polychaete', 'protist_star', 'hydromedusae_other', 'acantharia_protist', 'echinoderm_larva_pluteus_typeC', 'protist_dark_center', 'ctenophore_cydippid_no_tentacles', 'artifacts', 'siphonophore_physonect_young', 'artifacts_edge', 'diatom_chain_tube', 'trichodesmium_puff', 'trochophore_larvae', 'hydromedusae_partial_dark', 'pteropod_theco_dev_seq', 'siphonophore_calycophoran_abylidae', 'copepod_calanoid_large', 'ctenophore_cestid', 'echinoderm_seacucumber_auricularia_larva', 'unknown_sticks', 'hydromedusae_typeD', 'chaetognath_non_sagitta', 'radiolarian_colony', 'siphonophore_partial', 'hydromedusae_typeD_bell_and_tentacles', 'euphausiids_young', 'copepod_cyclopoid_oithona_eggs', 'invertebrate_larvae_other_B', 'siphonophore_other_parts', 'shrimp-like_other', 'stomatopod', 'copepod_calanoid', 'copepod_calanoid_flatheads', 'fish_larvae_medium_body', 'siphonophore_physonect', 'copepod_calanoid_frillyAntennae', 'amphipods', 'hydromedusae_haliscera', 'acantharia_protist_big_center', 'tunicate_salp', 'appendicularian_slight_curve', 'siphonophore_calycophoran_sphaeronectes_stem', 'hydromedusae_aglaura', 'fish_larvae_thin_body', 'hydromedusae_narco_young', 'radiolarian_chain', 'hydromedusae_narco_dark', 'shrimp_caridean', 'heteropod', 'copepod_cyclopoid_oithona', 'decapods', 'siphonophore_calycophoran_rocketship_young', 'ctenophore_cydippid_tentacles', 'copepod_calanoid_large_side_antennatucked', 'hydromedusae_solmaris', 'hydromedusae_h15', 'chordate_type1', 'shrimp_sergestidae', 'crustacean_other', 'diatom_chain_string', 'siphonophore_calycophoran_sphaeronectes', 'hydromedusae_narcomedusae', 'tunicate_doliolid', 'hydromedusae_shapeB', 'echinoderm_larva_pluteus_early', 'echinoderm_larva_pluteus_urchin', 'detritus_other']
    import os, sys
    if os.path.isfile('paper21'):
        stateIn = 'paper21'
    else:
        stateIn = None
    pred = LeNetPredictor(stateIn=stateIn)
    print("Loaded Predictor ")


    if (sys.platform == 'darwin'):
        #path = "/Users/oli/Proj_Large_Data/kaggle_plankton/test_resized/"
        path = "/Users/oli/Proj_Large_Data/kaggle_plankton/tmp/"
        path_training = "/Users/oli/Proj_Large_Data/kaggle_plankton/train_resized/"
        fout = open("/Users/oli/Proj_Large_Data/kaggle_plankton/submission.csv", 'w');
        fc = csv.reader(file('/Users/oli/Proj_Large_Data/kaggle_plankton/sampleSubmission.csv'))
    else:
        path_training = "/home/dueo/data_kaggel_bowl/train_resized/"
        path = "/home/dueo/data_kaggel_bowl/test_resized/"
        fout = open("/home/dueo/data_kaggel_bowl/submission.csv", 'w');
        fc = csv.reader(file('/home/dueo/data_kaggel_bowl/sampleSubmission.csv'))
    print " Using the following path " + str(path)



    #d = LoadPics.LoadPics(path_training)
    #print(d.getNumberOfClassed())


    files = os.listdir(path)
    try:
        files.remove('.DS_Store')
    except:
        pass

    c = 0
    import csv
    w = csv.writer(fout);

    #classes = d.getClasses()
    head = fc.next()

    c = -1
    idx = [-1] * len(classesasreadonserver)
    for sc in classesasreadonserver:
        c += 1
        print(c)
        for i, s in enumerate(head[1:]):
            if sc in s:
                idx[c] = i


    w.writerow(head)
    for fin in files:
        print (fin)
        pics = cv2.imread(path + fin , cv2.CV_LOAD_IMAGE_GRAYSCALE)
        X = np.reshape(pics / 255., len(pics)**2)
        res = pred.getPrediction(X)[0]
        idmax = np.argmax(res)
        name = classesasreadonserver[idmax]
        p = res[idmax]
        print("Name " + str(name) + " p " + str(p) + " pmax=" + str(res[59]))
        if (p > 0.05):
            print("Name " + str(name) + " p " + str(p))
            cv2.imshow('query', pics)
            cv2.waitKey(1000000)


        fout.write(fin + ',')
        w.writerow(res[idx])








