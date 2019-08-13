# %%
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_img(x):
    return preprocess_input(x, mode='tf')


#%%
