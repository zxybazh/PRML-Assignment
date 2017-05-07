# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from dataloader import load_data
import numpy as np
import cPickle as pkl


class GenerativeClassifier(object):
    """
        meta class for generative classifiers.
    """

    def __init__(self, preprocessing="", shuffle_train=False, shuffle_test=False):
        self.x_train, self.y_train, self.x_test, self.y_test, self.feature_size = load_data(t=preprocessing)
        self.train_size = len(self.y_train)
        self.test_size = len(self.x_test)

        # TODO: reset the prior distribution
        self.pi_1 = 0

        if shuffle_train:
            idx = np.arange(self.train_size)
            np.random.shuffle(idx)
            self.x_train = [self.x_train[idx[i]] for i in range(self.train_size)]
            self.y_train = [self.y_train[idx[i]] for i in range(self.train_size)]

        if shuffle_test:
            idx = np.arange(self.test_size)
            np.random.shuffle(idx)
            self.x_test = [self.x_test[idx[i]] for i in range(self.test_size)]
            self.y_test = [self.y_test[idx[i]] for i in range(self.test_size)]


"""
    this method using Beta Distribution as the conjuncture prior.
"""


class BetaNaiveBayesClassifier(GenerativeClassifier):

    def __init__(self, alpha = 0.5, preprocessing="binarized"):
        super(BetaNaiveBayesClassifier, self).__init__(preprocessing=preprocessing)
        self.alpha = alpha
        self.beta = alpha
        self.priorx = np.full((self.feature_size, 2), self.alpha, dtype=float)
        self.priory = np.full(2, 0, dtype=float)

    def train(self):
        for i in xrange(self.feature_size):
            x = zip(self.x_train[:,i], self.y_train)
            x_0 = np.array([w[0] for w in filter(lambda x: x[1] == 0, x)]);
            x_1 = np.array([w[0] for w in filter(lambda x: x[1] == 1, x)]);
            # Given x_0, x_1 are binarized we can sum up here
            self.priorx[i][0] = (sum(x_0) + self.alpha) / float(len(x_0) + self.alpha + self.beta);
            self.priorx[i][1] = (sum(x_1) + self.alpha) / float(len(x_1) + self.alpha + self.beta);
            self.priory[1] = sum(self.y_train)
            self.priory[0] = len(self.y_train) - self.priory[1]

    def test(self):
        self.count = [0, 0]
        self.ecount = [0, 0]
        for i in xrange(len(self.x_test)):
            x = self.x_test[i]
            y_0 = self.priory[0]
            y_1 = self.priory[1]
            for j in xrange(self.feature_size):
                y_0 *= x[j] * self.priorx[j][0] + (1 - x[j]) * (1 - self.priorx[j][0])
                y_1 *= x[j] * self.priorx[j][1] + (1 - x[j]) * (1 - self.priorx[j][1])
            if (y_0 > y_1): y = 0
            else: y = 1
            if y == self.y_test[i]:
                self.count[1] += 1
            else:
                self.count[0] += 1
                self.ecount[y] += 1
        print "-" * 20, "Beta Naive Bayes Classifier", "-" * 10
        print "Prior: Beta(", self.alpha, ",", self.alpha, ")"
        print "Correct Classcification:", self.count[1], ", Wrong Classcification:", self.count[0]
        print "Spam => Normal:", self.ecount[0], ", Normal => Spam:", self.ecount[1]
        ratio = 100 * self.count[1] / float(len(self.y_test))
        print "Correct Ratio: ", ratio, "%"
        print "-" * 59
        return ratio

class GaussianNaiveBayesClassifier(GenerativeClassifier):

    def __init__(self, preprocessing=""):
        super(GaussianNaiveBayesClassifier, self).__init__(preprocessing=preprocessing)
        self.ML = None

    def train(self):
        for i in xrange(self.feature_size):
            x = zip(self.x_train[:,i], self.y_train)
            x_0 = np.array([w[0] for w in filter(lambda x: x[1] == 0, x)]);
            x_1 = np.array([w[0] for w in filter(lambda x: x[1] == 1, x)]);
            
    def test(self):
        print "Gaussian Test >_<"

if __name__ == '__main__':
    Beta = BetaNaiveBayesClassifier();
    Beta.train()
    Beta.test()
    Gaussian = GaussianNaiveBayesClassifier()
    Gaussian.train()
    Gaussian.test()