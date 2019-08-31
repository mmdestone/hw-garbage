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

def preprocess_img(x):
    return preprocess_input(x, mode='torch')


# %%
with open(f'tmp/model_baseline-EfficientNet-B5.json', 'r') as f:
    model = model_from_json(f.read())
    model.load_weights(f'tmp/ckpt-baseline-EfficientNet-B5-Epoch_014-acc_0.97783-val_acc_0.92559.h5')


# %%
labels_valid = pd.read_csv('tmp/labels_valid.csv')
# labels_valid.label = labels_valid.label.apply(lambda x: f'{x:02d}')
# %%
lbs = []
for r in labels_valid.itertuples():
    img = Image.open('garbage_classify/train_data/' +
                     r.fname).resize((400, 400), Image.LANCZOS)
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
plt.figure(figsize=(9,9))
# plt.subplot(1, 2, 1)
# plt.matshow(mat)
# plt.colorbar()
# plt.subplot(1, 2, 2)
plt.imshow(mat/mat.sum(axis=1))
plt.colorbar()
# %%
plt.matshow((mat/mat.sum(axis=1)+mat.T/mat.sum(axis=1))/2)
plt.colorbar()

# %%
scores = np.diag(mat/labels_valid.groupby(by='label').count().values)
scores
# %%
label_scores = pd.DataFrame(
    {'scores': scores, 'label': [str(i) for i in range(40)]})
label_scores.plot(xticks=[i for i in range(40)],figsize=(16,9),grid=True,linewidth=3,style='-o')
# %%
