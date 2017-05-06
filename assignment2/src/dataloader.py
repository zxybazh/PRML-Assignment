import scipy.io as sio
import numpy as np
import math

def load_data(t="", preprocess=False, verbose=False, bias=False):
    data = sio.loadmat("../data/spamData.mat")
    feature_size = len(data["Xtrain"][1])

    def binarized(d):
        d = [1 if e > 0 else 0 for e in d]
        return np.array(d, dtype=int)

    def logtransform(d):
        d = [math.log(e+0.1) for e in d]
        return d

    def znormalization(d):
        # TODO: Please implement me.
        return d

    def flat(d):
        return np.array(d, dtype=int).flatten()

    Xtrain = data['Xtrain']
    ytrain = flat(data['ytrain'])
    Xtest = data['Xtest']
    ytest = flat(data['ytest'])

    if verbose:
        print "Loaded " + str(len(ytrain)) + " train samples, " + \
              "and " + str(len(ytest)) + " test samples, " + \
              "feature of X is of length " + str(feature_size) + "."

    # TODO: explore different preprocessing methods if needed.
    if t == "binarized":
        print "Binarized preprocessing!"
        return binarized(Xtrain), ytrain, binarized(Xtest), ytest, feature_size
    elif t == "log":
        print "log-transformation preprocessing!"
        return logtransform(Xtrain), ytrain, logtransform(Xtest), ytest, feature_size
    elif t == "z":
        print "z-normalization preprocessing!"
        return znormalization(Xtrain), ytrain, znormalization(Xtest), ytest, feature_size
    else:
        return Xtrain, ytrain, Xtest, ytest, feature_size


if __name__ == '__main__':
    load_data(verbose=True)