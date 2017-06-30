from dataloader import *
from discriminative_classfier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

# e_training = []
# e_test = []

# for Lambda in range(1, 10) + range(10, 105, 5):
# 	print "Processed to", Lambda
# 	logistic = LogisticRegression(l2norm=Lambda);
# 	acc_p, err_p = logistic.train()
# 	acc_q, err_q = logistic.test()
# 	e_training.append(acc_p)
# 	e_test.append(acc_q)

# file = open("../output/plot_training.out", "w")
# print >> file, e_training
# file.close()

# file1 = open("../output/plot_test.out", "w")
# print >> file1, e_test
# file1.close()

file = open("../output/plot_training.out", "r")
e_training = np.array([eval(line) for line in file.readlines()])[0]
file1 = open("../output/plot_test.out", "r")
e_test = np.array([eval(line) for line in file1.readlines()])[0]

data_lambda = range(1, 10) + range(10, 105, 5)

sns.set(style="darkgrid", color_codes=True)
#paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 1.2}
#sns.set_context("paper", rc = paper_rc)

df = pd.DataFrame(columns = ["Lambda", "Error Rate", "Error Type"])

for i in xrange(len(data_lambda)):
	d = {"Lambda": data_lambda[i], "Error Rate": e_training[i], "Error Type": "Training Error"}
	d = {"Lambda": data_lambda[i], "Error Rate": e_training[i], "Error Type": "Test Error"}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["Lambda", "Error Rate", "Error Type"]), ignore_index=True)

bar = sns.lmplot(x="Lambda", y="Error Rate", hue = "Error Type", data=df)

bar.set(xlabel='$\\lambda$', ylabel='Error Rate', title='Logistic Regressor Error Rate Change with $\\lambda$ value after 30 epochs')

plt.savefig('../output/foo.pdf')