from dataloader import *
from discriminative_classfier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random


file = open("../output/sgd.out", "r")
lines = file.readlines()
Recall_training    = np.array(eval(lines[0]))
Precision_training = np.array(eval(lines[1]))
Recall_test        = np.array(eval(lines[2]))
Precision_test     = np.array(eval(lines[3]))

sns.set(style="darkgrid", color_codes=True)
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1.2}
#sns.set_context("paper", rc = paper_rc)

df = pd.DataFrame(columns = ["Recall", "Precision", "Type"])

for i in xrange(len(Recall_training)):
	d  = {"Recall": Recall_training[i], "Precision": Precision_training[i], "Type": "Training"}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["Lambda", "Error Rate", "Type"]), ignore_index=True)

for i in xrange(len(Recall_test)):
	d  = {"Recall": Recall_test[i], "Precision": Precision_test[i], "Type": "Test"}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["Lambda", "Error Rate", "Type"]), ignore_index=True)

bar = sns.lmplot(x="Lambda", y="Error Rate", hue = "Type", data=df, legend = None)
bar.fig.get_axes()[0].legend(loc='upper right')
bar.set(xlabel='$\\lambda$', ylabel='Error Rate')

plt.savefig('../output/linear_'+pp+'.pdf')