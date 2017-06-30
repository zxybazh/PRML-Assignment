from dataloader import *
from discriminative_classifier import *

	logistic = LogisticRegression(l2norm=1);
	acc, err = logistic.train()
	acc, err = logistic.test()