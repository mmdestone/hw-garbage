# python -m src_yml.preprocess
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
# %%
path_data = 'garbage_classify/train_data/'
path_data_train = 'tmp/data_train/'
path_data_valid = 'tmp/data_valid/'
labels_file = 'tmp/labels_raw.csv'
img_size = 224
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
# # %%
# for r in tqdm(labels_val.itertuples(), desc='Validation', total=labels_val.shape[0]):
#     img = Image.open(path_data+r.fname)
#     img_new = img.resize((img_size, img_size), Image.BICUBIC)
#     img_new.save(path_data_valid+r.fname)
# # %%
# for r in tqdm(labels_tr.itertuples(), desc='Train raw', total=labels_tr.shape[0]):
#     img = Image.open(path_data+r.fname)
#     img_new = img.resize((img_size, img_size), Image.BICUBIC)
#     img_new.save(path_data_train+r.fname)
# %%
flips = [-1, Image.FLIP_LEFT_RIGHT]
rotates = [0, 30, 90, 120, 180, 210, 270, 300]
# filters = [ImageFilter.BoxBlur(
#     0), ImageFilter.GaussianBlur(7), ImageFilter.SHARPEN]
# ehcs = [ImageEnhance.Sharpness(1),
#         ImageEnhance.Sharpness(0.7),
#         # ImageEnhance.Sharpness(1.4),
#         ImageEnhance.Brightness(0.6),
#         ImageEnhance.Brightness(1.4),
#         ImageEnhance.Contrast(0.6),
#         ImageEnhance.Contrast(1.4),
#         ]
ehcs = [(ImageEnhance.Sharpness, 1, 's1'),
        (ImageEnhance.Sharpness, 0, 's0'),
        (ImageEnhance.Brightness, 0.5, 'b0.5'),
        (ImageEnhance.Brightness, 1.4, 'b1.4'),
        (ImageEnhance.Contrast, 0.5, 'c0.5'),
        (ImageEnhance.Contrast, 1.5, 'c1.5'),
        ]
# %%


def img_aug(r):
    data = []
    img_raw = Image.open(
        path_data+r.fname).resize((img_size, img_size), Image.BICUBIC)
    for ro in rotates:
        for flp in flips:
            for eh in ehcs:
                if (ro, flp, eh) == (0, -1, ehcs[0]):
                    continue
                fname = f'{ro}_{eh[2]}_{flp}_{r.fname}'
                data.append([fname, r.label])
                # 数据增强
                if flp == -1:
                    img_flp = img_raw
                else:
                    img_flp = img_raw.transpose(flp)
                img_rotate = img_flp.rotate(ro)
                img_ehc = eh[0](img_rotate).enhance(eh[1])
                img_ehc.save(path_data_train+fname)

    return pd.DataFrame(data, columns=labels.columns)


# %%
train_lbs = []
aug_times = len(rotates)*len(ehcs)*len(flips)  # 扩充倍数
min_smaples = labels_tr.groupby(by='label').count().min().fname
for label in labels.label.unique():
    label_imgs = labels_tr[labels_tr.label == label]
    c_samples = label_imgs.shape[0]
    alpha = (aug_times*min_smaples - c_samples)/((aug_times-1)*c_samples)
    alpha = alpha if alpha > 0 else 0
    print(f'Label {label:02d}, {c_samples:03d} samples, alpha={alpha:.4f}')
    imgs_aug = label_imgs.sample(frac=alpha)
    aug_rows = []
    for r in tqdm(imgs_aug.itertuples(), total=imgs_aug.shape[0]):
        new_rows = img_aug(r)
        aug_rows.append(new_rows)
        # break
    train_lbs.append(pd.concat(aug_rows))
    # break
new_lbs = pd.concat(train_lbs)
labels_train = pd.concat([labels_tr, new_lbs])  # 合并新旧数据
labels_train.to_csv('tmp/labels_train.csv', index=None)
# %%
labels_train = pd.read_csv('tmp/labels_train.csv')
labels_train.groupby(by='label').count().plot()


# %%
