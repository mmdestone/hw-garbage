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
with open(f'tmp/model_EfficientNet-B5-9.3.8-3.json', 'r') as f:
    model = model_from_json(f.read())
    model.load_weights(
        f'tmp/ckpt-EfficientNet-B5-9.3.8-3-Epoch_023-acc_0.99560-val_acc_0.94497.h5')

# %%
(b, w, h, c) = model.input_shape
batch_size = 16
# %%
labels_valid = pd.read_csv('tmp/labels_valid.csv')
labels_valid['lb'] = labels_valid.label.apply(lambda x: f'{x:02d}')

# %%
ig = ImageDataGenerator(preprocessing_function=efn.preprocess_input)

params_g = dict(
    batch_size=batch_size,
    # directory=path_data,
    # class_mode='other',
    x_col='fname',
    y_col='lb',
    target_size=(w, h),
    interpolation='lanczos',
    seed=201908)

valid_g = ig.flow_from_dataframe(
    labels_valid, 'garbage_classify/train_data', shuffle=False, **params_g)

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
%time model.evaluate_generator(valid_g, steps=valid_g.n//valid_g.batch_size)
# %%
preds = model.predict_generator(
    valid_g, steps=valid_g.n//valid_g.batch_size+1)

# %%
real_labels = labels_valid.label.values
pred_labels = np.argmax(preds, axis=1)
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
plt.figure(figsize=(9, 9))
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
scores = np.diag(
    mat/labels_valid[['label', 'fname']].groupby(by='label').count().values)
scores
# %%
label_scores = pd.DataFrame(
    {'scores': scores, 'label': [str(i) for i in range(40)]})
label_scores.plot(xticks=[i for i in range(40)], figsize=(
    16, 9), grid=True, linewidth=3, style='-o')
# %%
