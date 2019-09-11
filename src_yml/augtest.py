# %%
import glob
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
import efficientnet.keras as efn 

def preprocess_img(x):
    x = x / 127.5
    x -= 1.
    return x


keras.__version__

# %%
# img_size = 331
# model = load_model('tmp/ckpt.h5')
models = [None]*3
for i in range(len(models)):
    with open(f'tmp/model_{i}.json', 'r') as f:
        models[i] = model_from_json(f.read())
    models[i].load_weights(f'tmp/ckpt-{i}.h5')
# %%

# %%
img = Image.open('E:/garbage_classify/train_data/img_17725.jpg')

# %%


def aug_predict_multi_model(models, img0):
    res = []
    for model in models:
        img = img0.copy()
        (b, w, h, c) = model.input_shape
        img = img.resize((w, h))
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
        aug_imgs_arr = np.array([preprocess_img(np.array(x))
                                 for x in aug_imgs])
        res.append(model.predict(aug_imgs_arr))
    return np.array(res).sum(axis=(0, 1)).argmax()


# %%
%time res = aug_predict_multi_model(models, img)
res


# %%
print(res)
# %%
# %%
# 设置生成器参数
datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_img,
    channel_shift_range=30)

gen_data = datagen.flow_from_directory('garbage_classify',
                                       batch_size=9,
                                       shuffle=False,
                                       #    save_to_dir='tmp/test/',
                                       #    save_prefix='gen',
                                       target_size=(224, 224),
                                       interpolation='lanczos')


(img, label) = gen_data.next()


# plt.imshow(img[0])
# plt.imshow(img[1])
plt.imshow(img[2])
# %%
img = Image.open('E:/garbage_classify/train_data/img_17725.jpg')
img = Image.open('E:/garbage_classify/train_data/img_17615.jpg')
# img = Image.open('E:/garbage_classify/train_data/img_34.jpg')
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

    return np.array([preprocess_img(np.array(x)) for x in imgs_new])


# %%
with open(f'tmp/model_baseline-EfficientNet-B5.json', 'r') as f:
    model = model_from_json(f.read())
    model.load_weights(
        f'tmp/ckpt-baseline-EfficientNet-B5-Epoch_014-acc_0.97783-val_acc_0.92559.h5')

# %%
imgs = aug_images(img, (400, 400))
%time preds = model.predict(imgs)
np.argmax(preds, axis=1), np.array([preds]).sum(axis=(0, 1)).argmax()
# %%
