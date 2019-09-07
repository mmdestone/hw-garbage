# python -m src_yml.preprocess_v11
# %%
import numpy as np
import pandas as pd
from glob import glob
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, tqdm_gui, trange
from PIL import Image, ImageEnhance, ImageFilter
# %%
__version__ = 'v11'
path_data = 'garbage_classify/train_data_v2/'
path_data_train = f'tmp/data_train_{__version__}/'
path_data_valid = f'tmp/data_valid_{__version__}/'
path_data_train_extra = 'garbage_classify/train_data_extra/'
labels_file = 'tmp/labels_raw_v2.csv'
labels_file_extra = 'tmp/labels_extra.csv'
# %%

try:
    labels = pd.read_csv(labels_file)
except FileNotFoundError as e:
    print(e)
    labels = glob(f'{path_data}/*.txt')
    labels = pd.concat([pd.read_csv(label_f, header=None)
                        for label_f in labels])
    labels.columns = ['fname', 'label']
    labels.to_csv(labels_file, index=None)

labels.head()

# %%
kfold = StratifiedKFold(n_splits=5, random_state=201908, shuffle=True)

# %%
# for train, valid in kfold.split(labels.fname, labels.label):
#     print(train.shape, valid.shape)
# idx_tr, idx_val = next(kfold.split(labels.fname, labels.label))
# %%
for i, (idx_tr, idx_val) in enumerate(kfold.split(labels.fname, labels.label)):
    print(i)
    labels_tr = labels.iloc[idx_tr]
    labels_val = labels.iloc[idx_val]
    # %%
    labels_tr.to_csv(f'tmp/labels_train_{__version__}_fold{i}.csv', index=None)
    labels_val.to_csv(f'tmp/labels_valid_{__version__}_fold{i}.csv', index=None)

#%%
