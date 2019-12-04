
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
from sklearn.semi_supervised import LabelSpreading
# Ploting the result
c_arr = [0.0005,0.005,0.05,0.1,0.3,0.5,0.7,0.9,1,4]

evaAtk_suc_rate_sbst_arr = [1.0,1.0,1.0,1.0,1.0,1.0,0.96,0.56,0.22,0.01];
evaAtk_suc_rate_orac_arr =  [1.0,0.98,0.88,0.83,0.74,0.67,0.61,0.31,0.13,0.01];
tarAtk_suc_rate_sbst_arr =[1.0,1.0,1.0,1.0,0.999,0.98,0.71,0.139,0.036,0.001];
tarAtk_suc_rate_orac_arr =[0.591,0.363,0.183,0.149,0.099,0.073,0.044,0.007,0.001,0.001];

plt.figure()
plt.plot(c_arr,evaAtk_suc_rate_sbst_arr,'b.-')
plt.plot(c_arr,evaAtk_suc_rate_orac_arr,'go-')
plt.plot(c_arr,tarAtk_suc_rate_sbst_arr,'r^-')
plt.plot(c_arr,tarAtk_suc_rate_orac_arr,'kx-')
plt.legend(['evaAtk_sbst','evaAtk_orac','tarAtk_sbst','tarAtk_orac'])
plt.xlabel('c: the penalty coefficient')
plt.ylabel('Success Rate of Adversarial Attack')
plt.grid()
plt.show()