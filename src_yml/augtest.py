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

# %%
img_size = 299
model = load_model('tmp/ckpt.h5')
# %%
img = Image.open('E:/garbage_classify/train_data/img_17725.jpg').resize(
    (img_size, img_size), Image.LANCZOS)


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