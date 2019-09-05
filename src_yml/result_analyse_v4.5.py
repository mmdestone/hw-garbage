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
import math

# %%
with open(f'tmp/model_EfficientNet-B5-9.4.6-0.json', 'r') as f:
    model = model_from_json(f.read())
    model.load_weights(
        f'tmp/ckpt-EfficientNet-B5-9.4.6-0-Epoch_030-acc_0.99586-val_acc_0.94724.h5')

# %%
(b, w, h, c) = model.input_shape
batch_size = 16
# %%
labels_valid = pd.read_csv('tmp/labels_valid.csv')
labels_valid['lb'] = labels_valid.label.apply(lambda x: f'{x:02d}')
# %%
# 0.8888515782404298


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
    # imgs_flip = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgs]
    # imgs = imgs+imgs_flip
    imgs_new = []
    for img in imgs:
        imgs_new.append(img)
        # imgs_new.append(img.transpose(Image.ROTATE_90))
        # imgs_new.append(img.transpose(Image.ROTATE_180))
        # imgs_new.append(img.transpose(Image.ROTATE_270))

    return np.array([efn.preprocess_input(np.array(x)) for x in imgs_new])


lbs = []
for r in labels_valid.itertuples():
    img = Image.open('garbage_classify/train_data/' + r.fname)
    imgs = aug_images(img, (w, h))
    pred = model.predict(imgs)
    lbs.append(np.argmax(np.sum(pred, axis=0)))


real_labels = labels_valid.label.values
pred_labels = np.array(lbs)
pd.DataFrame(pred_labels).to_csv('tmp/preds.csv', index=False)
acc = (real_labels == pred_labels).sum()/real_labels.shape[0]
acc

# %%
real_labels = pd.read_csv('tmp/labels_valid.csv').label.values
pred_labels = pd.read_csv('tmp/preds.csv').values.flatten()


# %%
