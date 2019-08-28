#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#%%
labels = pd.read_csv('tmp/labels_raw.csv')

#%%
labels.groupby(by='label').count().plot(xticks=[i for i in range(40)],figsize=(16,9),grid=True,linewidth=3,style='-o')

#%%
