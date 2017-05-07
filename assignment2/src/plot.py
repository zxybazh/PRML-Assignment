import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random


sns.set(style="whitegrid", color_codes=True)
paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 1.2}
sns.set_context("paper", rc = paper_rc)   

file = open("../output/Beta.out", "r")
data1 = np.array([eval(line) for line in file.readlines()])

data_alpha = [i*0.5 for i in xrange(201)]
#print data_alpha

df = pd.DataFrame(columns = ["alpha", "Error rate"])

for i in xrange(len(data1)):
	d = {"alpha": data_alpha[i], "Error rate": data1[i]}
	df = df.append(pd.DataFrame(d, index = [0], columns = ["alpha", "Error rate"]), ignore_index=True)

bar = sns.regplot(x="alpha", y="Error rate", data=df, color = 'g')

bar.set(xlabel='$\\alpha$', ylabel='Time', title='Beta-Bernoulli Naive Bayes Model Error Rate Change with $\\alpha$ value')

plt.savefig('../output/foo.pdf')