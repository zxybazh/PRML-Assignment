import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
import random

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(15, 9))

norm = 7.0
data1 = np.array([11, 12, 13, 14])
data_config = [1, 2, 3, 4]
df1 = pd.DataFrame(columns = ["config", "time", "xxx"])
for i in xrange(4):
    d = {"config": data_config[i], "time": data1[i]/norm, "xxx": "ShMem%"}
    df1 = df1.append(pd.DataFrame(d, index = [0], columns = ["config", "time", "xxx"]), ignore_index=True)
df1

bar = sns.pointplot(x="config", y="time", hue = "xxx", data=df1, markers = "*")
for p in zip(bar.get_xticks(), df1["time"]):
    bar.text(p[0]-0.1, p[1]+0.2, str(p[1]*norm)+"%", color='#FFA500', fontsize=15)

bar.legend(loc='upper right')

bar.set(xlabel='$\alpha$', ylabel='Time', title='test title')
#plt.show()
plt.savefig('../output/foo.pdf')