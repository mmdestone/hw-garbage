import ast
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import OrderedDict
# from tensorflow.python.saved_model import tag_constants
from model_service.tfserving_model_service import TfServingBaseService
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

def preprocess_img(x):
    return preprocess_input(x, mode='tf')

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

class garbage_classify_service(TfServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 299  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        # config = tf.ConfigProto(allow_soft_placement=True)
        # with tf.get_default_graph().as_default():
        #     self.sess = tf.Session(graph=tf.Graph(), config=config)
        #     meta_graph_def = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_path)
        #     self.signature = meta_graph_def.signature_def

        #     # define input and out tensor of your model here
        #     input_images_tensor_name = self.signature[self.signature_key].inputs[self.input_key_1].name
        #     output_score_tensor_name = self.signature[self.signature_key].outputs[self.output_key_1].name
        #     self.input_images = self.sess.graph.get_tensor_by_name(input_images_tensor_name)
        #     self.output_score = self.sess.graph.get_tensor_by_name(output_score_tensor_name)

        self.model = load_model(self.model_path+'/ckpt.h5')

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

    # def center_img(self, img, size=None, fill_value=255):
    #     """
    #     center img in a square background
    #     """
    #     h, w = img.shape[:2]
    #     if size is None:
    #         size = max(h, w)
    #     shape = (size, size) + img.shape[2:]
    #     background = np.full(shape, fill_value, np.uint8)
    #     center_x = (size - w) // 2
    #     center_y = (size - h) // 2
    #     background[center_y:center_y + h, center_x:center_x + w] = img
    #     return background

    # def preprocess_img(self, img):
    #     """
    #     image preprocessing
    #     you can add your special preprocess method here
    #     """
    #     resize_scale = self.input_size / max(img.size[:2])
    #     img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    #     img = img.convert('RGB')
    #     img = np.array(img)
    #     img = img[:, :, ::-1]
    #     img = self.center_img(img, self.input_size)
    #     return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                # img = Image.open(file_content)
                img = Image.open(file_content)
                img = img.resize((self.input_size,self.input_size), Image.LANCZOS)
                # img = np.array(img)
                # img = preprocess_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        # img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        # # pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
        # pred_score = self.model.predict(img)
        # if pred_score is not None:
        #     pred_label = np.argmax(pred_score, axis=1)[0]
        #     result = {'result': self.label_id_name_dict[str(pred_label)]}
        # else:
        #     result = {'result': 'predict score is None'}
        pred_label= aug_predict(self.model,img)
        result = {'result': self.label_id_name_dict[str(pred_label)]}
        return result

    def _postprocess(self, data):
        return data
