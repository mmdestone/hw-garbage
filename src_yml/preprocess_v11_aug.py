# python -m src_yml.preprocess_v11_aug
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
path_data_aug = 'tmp/train_data_aug/'
labels_file = 'tmp/labels_raw_v2.csv'
fold = 0
labels_train = pd.read_csv(f'tmp/labels_train_{__version__}_fold{fold}.csv')
path_labels_train_aug = f'tmp/labels_train_aug_{__version__}_fold{fold}.csv'
# %%
shutil.rmtree(path_data_aug, True)
os.mkdir(path_data_aug)
# %%
rotates = [15, 30]
filters = [ImageFilter.GaussianBlur(2), ImageFilter.SHARPEN]
enhances = [(ImageEnhance.Brightness, 0.5),
            (ImageEnhance.Brightness, 1.4),
            (ImageEnhance.Contrast, 0.5),
            (ImageEnhance.Contrast, 1.4),
            (ImageEnhance.Color, 0.5),
            (ImageEnhance.Color, 1.4), ]
# %%
label_augs = []
for r in tqdm(labels_train.itertuples(), desc='Augmenting', total=labels_train.shape[0]):
    img_raw = Image.open(path_data+r.fname)
    img_raw.save(path_data_aug+r.fname)
    label_augs.append([r.fname, r.label])
    for rotate in rotates:
        fname = f'rotate_{rotate}_{r.fname}'
        label_augs.append([fname, r.label])
        img_raw.rotate(rotate).save(path_data_aug+fname)
    for i, flt in enumerate(filters):
        fname = f'filter_{i}_{r.fname}'
        label_augs.append([fname, r.label])
        img_raw.filter(flt).save(path_data_aug+fname)
    for i, ehc in enumerate(enhances):
        fname = f'enhance_{i}_{r.fname}'
        label_augs.append([fname, r.label])
        ehc[0](img_raw).enhance(ehc[1]).save(path_data_aug+fname)
    # break
# %%
label_augs = pd.DataFrame(label_augs, columns=['fname', 'label'])
label_augs.to_csv(path_labels_train_aug, index=None)
