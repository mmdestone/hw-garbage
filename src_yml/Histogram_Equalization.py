# %%
from PIL import Image
from skimage import data, exposure, color
import matplotlib.pyplot as plt
import numpy as np


# %%
img = np.array(Image.open(
    'E:/garbage_classify/train_data/img_17985.jpg').resize((331, 331), Image.BILINEAR))
img1 = exposure.equalize_adapthist(img)  # 进行自适应直方图均衡化
img2 = exposure.equalize_hist(img)

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Raw')
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title('adapthist')
plt.imshow(img1)
plt.subplot(1, 3, 3)
plt.title('hist')
plt.imshow(img2)


# %%
img3 = color.rgb2yuv(img)
img4 = img3.copy()
img4[:, :, 0] = exposure.equalize_adapthist(img3[:, :, 0])
img4 = color.yuv2rgb(img4)

img5 = img3.copy()
img5[:, :, 0] = exposure.equalize_hist(img3[:, :, 0])
img5 = color.yuv2rgb(img5)

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Raw')
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title('adapthist')
plt.imshow(img4)
plt.subplot(1, 3, 3)
plt.title('hist')
plt.imshow(img5)


# %%
img3 = color.rgb2yuv(img)
img4 = color.yuv2rgb(exposure.equalize_adapthist(img3))
img5 = color.yuv2rgb(exposure.equalize_hist(img3))

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Raw')
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title('adapthist')
plt.imshow(img4)
plt.subplot(1, 3, 3)
plt.title('hist')
plt.imshow(img5)

# %%


def to_square(img):
    (w, h) = img.size
    if h <= w:
        b = (w-h)//2
        box = (b, 0, h+b, h)
    else:
        b = (h-w)//2
        box = (0, b, w, w+b)
    return img.crop(box)

img = Image.open('E:/garbage_classify/train_data/img_17954.jpg')
img.show()
to_square(img).show()

# %%
img = Image.open('E:/garbage_classify/train_data/img_17954.jpg')

#%%
