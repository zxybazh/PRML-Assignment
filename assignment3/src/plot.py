from dataloader import *
from discriminative_classifier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

e_training = []
e_test = []

for Lambda in range(1, 10) + range(10, 105, 5):
	logistic = LogisticRegression(l2norm=Lambda);
	acc_p, err_p = logistic.train()
	acc_q, err_q = logistic.test()
	e_training.append(acc_p)
	e_test.append(acc_q)

data_lambda = range(1, 10) + range(10, 105, 5)

sns.set(style="darkgrid", color_codes=True)
paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 1.2}
sns.set_context("paper", rc = paper_rc)

df = pd.DataFrame(columns = ["lambda", "Training Error", "Test Error"])

for i in xrange(len(data1)):
	d = {"alpha": data_lambda[i], "Training Error": e_training[i], "Test Error": e_test[i]}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["lambda", "Training Error", "Test Error"]), ignore_index=True)

bar = sns.regplot(x="alpha", y="Error rate", data=df, color = 'g')

bar.set(xlabel='$\\lambda$', ylabel='Error rate', title='Logistic Regressor Error Rate Change with $\\lambda$ value after 100 epochs')

plt.savefig('../output/foo.pdf')