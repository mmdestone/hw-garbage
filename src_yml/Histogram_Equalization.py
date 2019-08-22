# %%
from PIL import Image
from skimage import data, exposure
import matplotlib.pyplot as plt
import numpy as np


# %%
img = Image.open(
    'E:/garbage_classify/train_data/img_17984.jpg').resize((331, 331), Image.BILINEAR)
img1 = exposure.equalize_adapthist(np.array(img))  # 进行自适应直方图均衡化
img2 = exposure.equalize_hist(np.array(img))

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


#%%
