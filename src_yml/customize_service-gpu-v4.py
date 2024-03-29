import ast
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import OrderedDict
# from tensorflow.python.saved_model import tag_constants
from model_service.caffe_model_service import CaffeBaseService

import keras
from keras.models import *


def preprocess_img(x):
    x = x / 127.5
    x -= 1.
    return x


def to_square(img):
    m = max(img.size)
    return img.resize((m, m))


def aug_predict(model, img):
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug_imgs = [img, img_flip,
                img.transpose(Image.ROTATE_90),
                img.transpose(Image.ROTATE_180),
                img.transpose(Image.ROTATE_270),
                img_flip.transpose(Image.ROTATE_90),
                img_flip.transpose(Image.ROTATE_180),
                img_flip.transpose(Image.ROTATE_270)]
    aug_imgs_arr = np.array([preprocess_img(np.array(x)) for x in aug_imgs])
    res = model.predict(aug_imgs_arr)
    lbs = np.argmax(res, axis=1).tolist()
    return max(set(lbs), key=lbs.count)


class garbage_classify_service(CaffeBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        if self.check_tf_version() is False:
            raise Exception('current use tensorflow CPU version')
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        # self.input_size = 331  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'

        # self.model = load_model(self.model_path+'/ckpt.h5')
        with open(self.model_path+'/model.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(self.model_path+'/ckpt.h5')

        (b, self.input_width, self.input_height, c) = self.model.input_shape

        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }

    def check_tf_version(self):
        from tensorflow.python.client import device_lib
        is_gpu_version = False
        devices_info = device_lib.list_local_devices()
        for device in devices_info:
            if 'GPU' == str(device.device_type):
                is_gpu_version = True
                break
        if is_gpu_version:
            print('use tensorflow-gpu', tf.__version__)
        else:
            print('use tensorflow', tf.__version__)
        print('Keras version:', keras.__version__)
        return is_gpu_version

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = to_square(Image.open(file_content))
                img = img.resize(
                    (self.input_width, self.input_height), Image.LANCZOS)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        pred_label = aug_predict(self.model, img)
        result = {'result': self.label_id_name_dict[str(pred_label)]}
        return result

    def _postprocess(self, data):
        return data
