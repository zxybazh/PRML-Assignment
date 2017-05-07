import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

sns.set(style="whitegrid", color_codes=True)
file = open("../output/Beta.out", "r")
data1 = np.array([eval(line) for line in file.readlines()])

data_alpha = [i*0.5 for i in xrange(201)]
print data_alpha

df = pd.DataFrame(columns = ["alpha", "time", "xxx"])
for i in xrange(4):
    d = {"alpha": data_alpha[i], "time": data1[i], "xxx": "ShMem%"}
    df1 = df1.append(pd.DataFrame(d, index = [0], columns = ["alpha", "time", "xxx"]), ignore_index=True)
df1

bar = sns.pointplot(x="alpha", y="time", hue = "xxx", data=df1, markers = "*")

bar.legend(loc='upper right')
bar.set(xlabel='\alpha', ylabel='Time', title='test title')
plt.savefig('../output/foo.pdf')