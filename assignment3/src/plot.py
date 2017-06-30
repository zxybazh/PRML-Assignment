from dataloader import *
from discriminative_classifier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

sns.set(style="darkgrid", color_codes=True)
paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 1.2}
sns.set_context("paper", rc = paper_rc)  

for Lambda in range(1, 10) + range(10, 105, 5):
	logistic = LogisticRegression(l2norm=Lambda);
	acc_p, err_p = logistic.train()
	acc_q, err_q = logistic.test()
	
	file = open("../output/Beta.out", "r")
	data1 = np.array([eval(line) for line in file.readlines()])
	file1 = open("../output/Beta_train.out", "r")
	data2 = np.array([eval(line) for line in file1.readlines()])

	data_alpha = [i*0.5 for i in xrange(201)]
	#print data_alpha

	df = pd.DataFrame(columns = ["alpha", "Error rate"])

	for i in xrange(len(data1)):
		d = {"alpha": data_alpha[i], "Error rate": data1[i]}
		df = df.append(pd.DataFrame(d, index = [0], columns = ["alpha", "Error rate"]), ignore_index=True)

	bar = sns.regplot(x="alpha", y="Error rate", data=df, color = 'g')

	df = pd.DataFrame(columns = ["alpha", "Error rate"])

	for i in xrange(len(data1)):
		d = {"alpha": data_alpha[i], "Error rate": data2[i]}
		df = df.append(pd.DataFrame(d, index = [0], columns = ["alpha", "Error rate"]), ignore_index=True)

	bar = sns.regplot(x="alpha", y="Error rate", data=df, color = 'r')

	bar.set(xlabel='$\\alpha$', ylabel='Error rate', title='Beta-Bernoulli Naive Bayes Model Error Rate Change with $\\alpha$ value')

	plt.savefig('../output/foo.pdf')