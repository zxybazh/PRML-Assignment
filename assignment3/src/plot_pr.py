from dataloader import *
from discriminative_classfier import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

linear = LinearRegression(miniBatch = 256)
eve = linear.train()
Precision_training = np.array(eve[0])
Recall_training    = np.array(eve[1])
Precision_test     = np.array(eve[2])
Recall_test        = np.array(eve[3])

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

plt.plot(Recall_training, Precision_training, label='Precision-Recall curve')
plt.show()

plt.plot(Recall_test, Precision_test, label='Precision-Recall curve')
plt.show()