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
from keras.models import Model, Sequential, Input
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import PIL
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %%
path_data = 'E:/garbage_classify/train_data/'


# %%
labels = glob(f'{path_data}/*.txt')
labels = pd.concat([pd.read_csv(label_f, header=None) for label_f in labels])


# %%
labels.columns = ['fname', 'label']
labels.label = labels.label.apply(str)
labels.head()


# %%
ig = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2)

batch_size = 64

train_g = ig.flow_from_dataframe(
    labels, path_data, batch_size=batch_size, x_col='fname', y_col='label', target_size=(224, 224), subset='training')
valid_g = ig.flow_from_dataframe(
    labels, path_data, batch_size=batch_size, x_col='fname', y_col='label', target_size=(224, 224), subset='validation')


# %%
n_classess = labels.label.unique().shape[0]
n_classess


# %%
base_model = InceptionV3(weights='imagenet', include_top=False)
# base_model = InceptionV3(weights=None, include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有200个类
predictions = Dense(n_classess, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False
# %%
ckpt = ModelCheckpoint(
    'tmp/ckpt-'+time.strftime('%Y-%m-%d_%H_%M')+'-Epoch_{epoch:03d}-acc_{acc:.5f}-val_acc_{val_acc:.5f}.h5', save_best_only=True, monitor='val_acc')
estop = EarlyStopping(monitor='val_acc', min_delta=1e-7,
                      verbose=1, patience=20)

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(
    train_g,
    # steps_per_epoch=1,
    steps_per_epoch=train_g.n // batch_size,
    epochs=100,
    callbacks=[ckpt, estop],
    validation_data=valid_g,
    # validation_steps=1,
    validation_steps=valid_g.n // batch_size
)
