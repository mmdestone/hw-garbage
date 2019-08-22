# %%
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model
from keras.applications.nasnet import NASNetLarge
from keras.models import Model, Sequential, Input
from keras.layers import *
from keras.optimizers import Adam,SGD

def preprocess_img(x):
    x = x / 127.5
    x -= 1.
    return x

keras.__version__
#%%
img_size = 331
img_width = img_size
img_height = img_size
def get_model():
    base_model = NASNetLarge(
    weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x=Dropout(0.3)(x)

    predictions = Dense(40, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
# %%
# model = load_model('tmp/ckpt.h5')
model = get_model()
model.load_weights('tmp/ckpt.h5')
# %%
img = Image.open('E:/garbage_classify/train_data/img_17725.jpg').resize(
    (331, 331), Image.LANCZOS)


# %%
def aug_predict(model, img):
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug_imgs = [
        img, img_flip,
        img.transpose(Image.ROTATE_90),
        img.transpose(Image.ROTATE_180),
        img.transpose(Image.ROTATE_270),
        img_flip.transpose(Image.ROTATE_90),
        img_flip.transpose(Image.ROTATE_180),
        img_flip.transpose(Image.ROTATE_270)
    ]
    aug_imgs_arr = np.array([preprocess_img(np.array(x)) for x in aug_imgs])
    # aug_imgs_arr = np.array([np.array(x) for x in aug_imgs])
    # aug_imgs_arr = preprocess_img(aug_imgs_arr)

    res = model.predict(aug_imgs_arr, batch_size=8)
    lbs = np.argmax(res, axis=1).tolist()
    print(lbs)
    return max(set(lbs), key=lbs.count)


# %%
%time res = aug_predict(model, img)
res


# %%
print(res)