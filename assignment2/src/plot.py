import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random


sns.set(style="whitegrid", color_codes=True)
paper_rc = {'lines.linewidth': 0.4, 'lines.markersize': 0.8}
sns.set_context("paper", rc = paper_rc)   

file = open("../output/Beta.out", "r")
data1 = np.array([eval(line) for line in file.readlines()])

data_alpha = [i*0.5 for i in xrange(201)]
#print data_alpha

df = pd.DataFrame(columns = ["alpha", "Error rate", "xxx"])
for i in xrange(len(data1)):
	if (i % 10 != 0):
		data_alpha[i] = '';
	else:
		data_alpha[i] = str(data_alpha[i])
	d = {"alpha": data_alpha[i], "Error rate": data1[i], "xxx": "Beta-Bernoulli Naive Bayes"}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["alpha", "Error rate", "xxx"]), ignore_index=True)

bar = sns.pointplot(x="alpha", y="Error rate", hue = "xxx", data=df)

bar.legend(loc='upper right')
bar.set(xlabel='$\\alpha$', ylabel='Time', title='Beta-Bernoulli Naive Bayes Model Error Rate Change with $\\alpha$ value')
plt.savefig('../output/foo.pdf')