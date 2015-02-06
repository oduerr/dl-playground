__author__ = 'oli'

# See also https://github.com/jameslyons/python_speech_features


from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("/Users/oli/Proj_Large_Data/KuhKauen/versionControl/kuh/experimental/kuh_wiederkauen.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print mfcc_feat.shape
print fbank_feat.shape