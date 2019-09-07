# %%
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import *
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
import efficientnet.keras as efn
from keras.metrics import categorical_accuracy
from tqdm import tqdm

# %%
with open(f'tmp/model_EfficientNet-B5-9.5.14-2.json', 'r') as f:
    model = model_from_json(f.read())
    model.load_weights(
        f'tmp/ckpt-EfficientNet-B5-9.5.14-2-Epoch_063-acc_0.99573-val_acc_0.94769_compressed.h5')

# %%
(b, w, h, c) = model.input_shape
batch_size = 16
# %%
labels_valid = pd.read_csv('tmp/labels_valid_v11_fold0.csv')
labels_valid['lb'] = labels_valid.label.apply(lambda x: f'{x:02d}')
# %%


def aug_images(img_raw, img_size=(299, 299)):
    (w, h) = img_raw.size
    if h <= w:
        b = (w-h)//2
        box_center = (b, 0, h+b, h)
        box_top = (0, 0, h, h)
        box_bottom = (w-h, 0, w, h)
    else:
        b = (h-w)//2
        box_center = (0, b, w, w+b)
        box_top = (0, 0, w, w)
        box_bottom = (0, h-w, w, h)

    imgs = [
        img_raw.resize(img_size, Image.LANCZOS),
        img_raw.crop(box_center).resize(img_size, Image.LANCZOS),
        img_raw.crop(box_top).resize(img_size, Image.LANCZOS),
        img_raw.crop(box_bottom).resize(img_size, Image.LANCZOS),
    ]
    imgs_flip = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgs]
    imgs = imgs+imgs_flip
    imgs_new = []
    for img in imgs:
        imgs_new.append(img)
        imgs_new.append(img.transpose(Image.ROTATE_90))
        imgs_new.append(img.transpose(Image.ROTATE_180))
        imgs_new.append(img.transpose(Image.ROTATE_270))

    return np.array([efn.preprocess_input(np.array(x)) for x in imgs_new])


# %%
lbs = []
for r in tqdm(labels_valid.itertuples(), total=labels_valid.shape[0]):
    img = Image.open('garbage_classify/train_data_v2/' + r.fname)
    imgs = aug_images(img, (w, h))
    pred = model.predict(imgs)
    lbs.append(pred)
# %%
lbs_np = np.array(lbs)
np.save('tmp/pred_lbs.npy', lbs_np)
# %%
lbs_np = np.load('tmp/pred_lbs.npy')
lbs_np.shape
# %%
