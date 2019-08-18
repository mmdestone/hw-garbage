# python -m src_yml.preprocess_v4
# %%
try:
    import warnings
    warnings.filterwarnings('ignore')
    import utils
except Exception as e:
    print(e)
    pass
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
from concurrent.futures import wait, ProcessPoolExecutor
# %%
path_data = 'garbage_classify/train_data/'
path_data_train = 'tmp/data_train/'
path_data_valid = 'tmp/data_valid/'
labels_file = 'tmp/labels_raw.csv'
img_size = 224
samples_per_label = 6000

# %%
flips = [-1, Image.FLIP_LEFT_RIGHT]
rotates = [0, 30, 90, 120, 180, 210, 270, 300]
filters = [ImageFilter.BoxBlur(
    0), ImageFilter.GaussianBlur(2), ImageFilter.SHARPEN]
enhances = [(ImageEnhance.Brightness, 1),
            (ImageEnhance.Brightness, 0.5),
            (ImageEnhance.Brightness, 1.4),
            (ImageEnhance.Contrast, 0.5),
            (ImageEnhance.Contrast, 1.4),
            (ImageEnhance.Color, 0),
            (ImageEnhance.Color, 0.5),
            (ImageEnhance.Color, 1.5),
            ]
# %%


def handle(label, augs_df, labels_tr):
    labels_tr_aug = []
    label_imgs = augs_df[augs_df.label == label]
    n_samples = (labels_tr.label == label).sum()

    imgs_aug = label_imgs.sample(n=samples_per_label - n_samples,
                                 random_state=201908)
    print(
        f'Label {label:2d},n_samples = {n_samples},new_samples = {imgs_aug.shape[0]}')
    for r in imgs_aug.itertuples():
        fname = f'{r.flp}_{r.ro}_{r.flt}_{r.ehc}_{r.fname}'
        labels_tr_aug.append([fname, r.label])
        img_raw = Image.open(
            path_data+r.fname).resize((img_size, img_size), Image.LANCZOS)
        if r.flp == 0:
            img_flp = img_raw
        else:
            img_flp = img_raw.transpose(flips[r.flp])
        img_rotate = img_flp.rotate(rotates[r.ro])
        img_flt = img_rotate.filter(filters[r.flt])
        eh = enhances[r.ehc]
        img_ehc = eh[0](img_flt).enhance(eh[1])
        img_ehc.save(path_data_train+fname)
        # break
    # break
    print(f'Label {label:2d} done.')
    return pd.DataFrame(labels_tr_aug, columns=['fname', 'label'])

if __name__ == '__main__':
    # %%
    print('New Process.')

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
    idx_tr, idx_val = next(kfold.split(labels.fname, labels.label))
    # %%
    labels_tr = labels.iloc[idx_tr]
    labels_val = labels.iloc[idx_val]
    # labels_tr.to_csv('tmp/labels_train.csv',index=None)
    labels_val.to_csv('tmp/labels_valid.csv', index=None)

    # %%
    labels_tr.groupby(by='label').count().plot()

    # %%
    labels_val.groupby(by='label').count().plot()
    # %%
    shutil.rmtree(path_data_train, True)
    shutil.rmtree(path_data_valid, True)
    os.mkdir(path_data_train)
    os.mkdir(path_data_valid)
    # %%
    for r in tqdm(labels_val.itertuples(), desc='Validation', total=labels_val.shape[0]):
        img = Image.open(path_data+r.fname)
        img_new = img.resize((img_size, img_size), Image.LANCZOS)
        img_new.save(path_data_valid+r.fname)
    # %%
    for r in tqdm(labels_tr.itertuples(), desc='Train raw', total=labels_tr.shape[0]):
        img = Image.open(path_data+r.fname)
        img_new = img.resize((img_size, img_size), Image.LANCZOS)
        img_new.save(path_data_train+r.fname)

    # %%
    augs = []
    for r in tqdm(labels_tr.itertuples(), desc='Combining', total=labels_tr.shape[0]):
        for flp in range(len(flips)):
            for ro in range(len(rotates)):
                for flt in range(len(filters)):
                    for ehc in range(len(enhances)):
                        if (flp, ro, flt, ehc) == (0, 0, 0, 0):
                            continue
                        else:
                            augs.append([r.fname, r.label, flp, ro, flt, ehc])

    # %%
    augs_df = pd.DataFrame(
        augs, columns=['fname', 'label', 'flp', 'ro', 'flt', 'ehc'])
    # %%


    # %%
    pool = ProcessPoolExecutor()
    tds = []
    for label in labels.label.unique():
        td = pool.submit(handle, label, augs_df, labels_tr)
        tds.append(td)
    wait(tds)
    labels_tr_aug = pd.concat([td.result() for td in tds])
    labels_train = pd.concat([labels_tr, labels_tr_aug])
    labels_train.to_csv('tmp/labels_train.csv', index=None)
    # %%
    labels_train = pd.read_csv('tmp/labels_train.csv')
    labels_train.groupby(by='label').count().plot()
    # %%
    print(labels_tr_aug.shape,labels_train.shape)
