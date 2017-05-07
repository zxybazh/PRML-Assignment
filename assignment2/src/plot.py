import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from matplotlib import style
import random

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(15, 9))

#-------------------- bar plot for RNNs ----------------------------------------------#

norm = 7.0
data1 = np.array([12.3, 29.7, 45.7, 50.3, 55.0])
data_config = [" 1%(28SMs)", " 5%(56SMs)", "10%(56SMs)", "20%(56SMs)", "30%(56SMs)"]
df1 = pd.DataFrame(columns = ["config", "percent", "xxx"])
for i in xrange(5):
    d = {"config": data_config[i], "percent": data1[i]/norm, "xxx": "ShMem%"}
    df1 = df1.append(pd.DataFrame(d, index = [0], columns = ["config", "percent", "xxx"]), ignore_index=True)
df1
#-------------------- data for pointplot (SHM percent) -------------------------------#

bar1 = sns.pointplot(x="config", y="percent", hue = "xxx", data=df1, markers = "*")
for p in zip(bar1.get_xticks(), df1["percent"]):
    bar1.text(p[0]-0.1, p[1]+0.2, str(p[1]*norm)+"%", color='#FFA500', fontsize=15)

#-------------------- Print Plot -----------------------------------------------------#
bar1.legend(loc='upper right')

bar.set(xlabel='nonzero% (SM# used by sparsePersistentRNN)', ylabel='Speedup(vs cudnnRNN)', \
        title='(a) hiddensize 1792, batchsize 4, timestep 256, varying nonzero%')
#plt.show()
plt.savefig('foo1.pdf')