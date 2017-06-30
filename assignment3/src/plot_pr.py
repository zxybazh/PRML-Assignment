from dataloader import *
from discriminative_classfier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random


file = open("../output/linear_plot_training.out", "r")
lines = file.readlines()
Recall_training    = np.array([eval(line) for line in file.readlines()])
Precision_training = np.array([eval(line) for line in file.readlines()])[
Recall_test        = np.array([eval(line) for line in file.readlines()])[0]
Precision_test     = np.array([eval(line) for line in file.readlines()])[0]
# file1 = open("../output/linear_plot_test.out", "r")
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

plt.savefig('../output/linear_'+pp+'.pdf')