# python -m src_yml.preprocess_v12
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
__version__ = 'v10'
path_data = 'garbage_classify/train_data_v2/'
path_data_train = f'tmp/data_train_{__version__}/'
path_data_valid = f'tmp/data_valid_{__version__}/'
path_data_train_extra = 'garbage_classify/train_data_ext/'
labels_file = 'tmp/labels_raw_v2.csv'
labels_file_extra = 'tmp/labels_ext.csv'
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
try:
    labels_extra = pd.read_csv(labels_file_extra)
except FileNotFoundError as e:
    print(e)
    labels_extra = glob(f'{path_data_train_extra}/*.txt')

    labels_extra = pd.concat([pd.read_csv(label_f, header=None)
                              for label_f in labels_extra])
    labels_extra.columns = ['fname', 'label']
    labels_extra.to_csv(labels_file_extra, index=None)

labels_extra.head()
# %%
kfold = StratifiedKFold(n_splits=5, random_state=201908, shuffle=True)

# %%
# for train, valid in kfold.split(labels.fname, labels.label):
#     print(train.shape, valid.shape)
idx_tr, idx_val = next(kfold.split(labels.fname, labels.label))
# %%
labels_tr = labels.iloc[idx_tr]
labels_val = labels.iloc[idx_val]
labels_val.head()
# %%
labels_tr.to_csv(f'tmp/labels_train_{__version__}.csv', index=None)
labels_val.to_csv(f'tmp/labels_valid_{__version__}.csv', index=None)
# %%
shutil.rmtree(path_data_train, True)
shutil.rmtree(path_data_valid, True)
os.mkdir(path_data_train)
os.mkdir(path_data_valid)
# # %%
# for r in tqdm(labels_val.itertuples(), desc='Validation', total=labels_val.shape[0]):
#     # img = Image.open(path_data+r.fname)
#     # # img_new = img.resize((img_size, img_size), Image.LANCZOS)
#     # img.save(path_data_valid+r.fname)
#     shutil.copyfile(path_data+r.fname, path_data_valid+r.fname)
# # %%
# for r in tqdm(labels_tr.itertuples(), desc='Train', total=labels_tr.shape[0]):
#     # img = Image.open(path_data+r.fname)
#     # # img_new = img.resize((img_size, img_size), Image.LANCZOS)
#     # img.save(path_data_train+r.fname)
#     shutil.copyfile(path_data+r.fname, path_data_train+r.fname)
# %%
for r in tqdm(labels_extra.itertuples(), desc='Train_extra', total=labels_extra.shape[0]):
    # img = Image.open(path_data+r.fname)
    # # img_new = img.resize((img_size, img_size), Image.LANCZOS)
    # img.save(path_data_train+r.fname)
    shutil.copyfile(path_data_train_extra + r.fname, path_data+r.fname)


#%%
