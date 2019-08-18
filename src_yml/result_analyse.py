# %%
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input


def preprocess_img(x):
    return preprocess_input(x, mode='tf')


# %%
model = load_model('tmp/ckpt.h5')


# %%
labels_valid = pd.read_csv('tmp/labels_valid.csv')
# labels_valid.label = labels_valid.label.apply(lambda x: f'{x:02d}')
# %%
lbs = []
for r in labels_valid.itertuples():
    img = Image.open('garbage_classify/train_data/' +
                     r.fname).resize((224, 224), Image.LANCZOS)
    img = np.array(img)
    img = preprocess_img(img)
    pred = model.predict(np.array([img]))
    lbs.append(np.argmax(pred, axis=1)[0])

# %%
real_labels = labels_valid.label.values
pred_labels = np.array(lbs)
pd.DataFrame(pred_labels).to_csv('tmp/preds.csv', index=False)
# %%
acc = (real_labels == pred_labels).sum()/real_labels.shape[0]
acc

# %%
real_labels = pd.read_csv('tmp/labels_valid.csv').label.values
pred_labels = pd.read_csv('tmp/preds.csv').values.flatten()

# %%
mat = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        mat[i, j] = ((real_labels == i) & (pred_labels == j)).sum()


# %%
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.matshow(mat)
# plt.colorbar()
# plt.subplot(1, 2, 2)
plt.matshow(mat/mat.sum(axis=1))
plt.colorbar()
# %%