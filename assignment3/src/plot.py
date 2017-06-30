from dataloader import *
from discriminative_classifier import *

for Lambda in range(1, 10) + range(10, 105, 5):
	logistic = LogisticRegression(l2norm=Lambda);
	acc, err = logistic.train()
	acc, err = logistic.test()