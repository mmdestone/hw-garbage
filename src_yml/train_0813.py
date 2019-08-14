# python -m src_yml.train_0813
# %%
# !pip install  tensorflow-gpu==1.13.1 pillow pandas matplotlib keras pillow
# %%
try:
    import warnings
    warnings.filterwarnings('ignore')
    import utils
except Exception as e:
    print(e)
    pass

# %%
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential, Input
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import PIL
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
path_data = 'garbage_classify/train_data'
batch_size = 16
img_width = 224
img_height = 224
img_size = 224
random_seed = 201908
path_data_train = 'tmp/data_train/'
path_data_valid = 'tmp/data_valid/'
labels_file = 'tmp/labels_raw.csv'


# %%
labels_train = pd.read_csv('tmp/labels_train.csv')
labels_valid = pd.read_csv('tmp/labels_valid.csv')
n_classess = labels_train.label.unique().shape[0]
n_classess
labels_train.groupby(by='label').count().plot()
# %%
labels_train.label = labels_train.label.apply(lambda x: f'{x:02d}')
labels_valid.label = labels_valid.label.apply(lambda x: f'{x:02d}')
# labels_train['label_bin'].values = keras.utils.np_utils.to_categorical(
#     labels_train.label, n_classess)
# %%
ig = ImageDataGenerator(preprocessing_function=utils.preprocess_img)

params_g = dict(
    batch_size=batch_size,
    # directory=path_data,
    # class_mode='other',
    x_col='fname',
    y_col='label',
    target_size=(img_width, img_height),
    seed=random_seed)

train_g = ig.flow_from_dataframe(
    labels_train, path_data_train, **params_g)
valid_g = ig.flow_from_dataframe(
    labels_valid, path_data_valid, **params_g)
# %%
# (imgs, lbs) = next(train_g)
# imgs.shape, lbs.shape


# %%
base_model = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# base_model = InceptionV3(weights=None, include_top=False)
# base_model = ResNet50(
#     weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
# x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)

# 添加一个分类器，假设我们有200个类
predictions = Dense(n_classess, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)
# model.summary()
# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
# for layer in base_model.layers:
#     layer.trainable = False
# %%
ckpt = ModelCheckpoint(
    'tmp/ckpt-'+time.strftime('%Y-%m-%d_%H_%M')+'-Epoch_{epoch:03d}-acc_{acc:.5f}-val_acc_{val_acc:.5f}.h5', save_best_only=True, monitor='val_acc')
estop = EarlyStopping(monitor='val_acc', min_delta=1e-7,
                      verbose=1, patience=20)

# %%
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(
    train_g,
    # steps_per_epoch=100,
    steps_per_epoch=train_g.n // batch_size,
    epochs=100,
    callbacks=[ckpt, estop],
    validation_data=valid_g,
    # validation_steps=1,
    validation_steps=valid_g.n // batch_size
)
