from dataloader import *
from discriminative_classfier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

data_lambda = range(1, 10) + range(10, 105, 5)
for pp in ["", "z", "log", "binarized"]:
	e_training = []
	e_test = []
	for Lambda in range(1, 10) + range(10, 105, 5):
		#print "Processed to", Lambda
		logistic = LogisticRegression(preprocessing=pp, max_epoch=100, l2norm=Lambda);
		acc_p, err_p = logistic.train()
		acc_q, err_q = logistic.test()
		if (Lambda in [1, 10, 100]):
			print "preprocessing:", pp, "Lambda:", Lambda, "acc_training:", acc_p, "acc_test:", acc_q
		e_training.append(acc_p)
		e_test.append(acc_q)

	file = open("../output/plot_training_"+pp+".out", "w")
	print >> file, e_training
	file.close()

	file1 = open("../output/plot_test_"+pp+".out", "w")
	print >> file1, e_test
	file1.close()

	# file = open("../output/plot_training.out", "r")
	# e_training = np.array([eval(line) for line in file.readlines()])[0]
	# file1 = open("../output/plot_test.out", "r")
	# e_test = np.array([eval(line) for line in file1.readlines()])[0]

	sns.set(style="darkgrid", color_codes=True)
	paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1.2}
	#sns.set_context("paper", rc = paper_rc)

	df = pd.DataFrame(columns = ["Lambda", "Error Rate", "Type"])


	for i in xrange(len(data_lambda)):
		d  = {"Lambda": data_lambda[i], "Error Rate": e_training[i], "Type": "Training Error"}
		d1 = {"Lambda": data_lambda[i], "Error Rate": e_test[i], "Type": "Test Error"}
		df = df.append(pd.DataFrame(d , index = [0], columns = ["Lambda", "Error Rate", "Type"]), ignore_index=True)
		df = df.append(pd.DataFrame(d1, index = [0], columns = ["Lambda", "Error Rate", "Type"]), ignore_index=True)


	bar = sns.lmplot(x="Lambda", y="Error Rate", hue = "Type", data=df, legend = None)
	bar.fig.get_axes()[0].legend(loc='upper right')
	bar.set(xlabel='$\\lambda$', ylabel='Error Rate')

	plt.savefig('../output/foo_'+pp+'.pdf')