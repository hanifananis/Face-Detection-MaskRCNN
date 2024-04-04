# coding:utf-8
import os
import sys
import numpy as np
import skimage.io
import skimage.draw
import json
from config import Config
import utils
import model as modellib

class DataSetConfig(Config):
    """
    数据集配置，继承自Mask R-CNN Config
    """
    GPU_COUNT = 1
    NAME = "FaceDetection"
    IMAGE_MAX_DIM = 1024
    IMAGES_PER_GPU = 1  # GPU批量处理的图片个数
    NUM_CLASSES = 1 + 1  # GPU处理的类别（background + face）
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class FaceDataSet(utils.Dataset):

    def load_face(self, dataset_dir):
        """
        从人脸数据集中加载数据
        :param dataset_dir: 数据集bounding box位置
        """
        self.add_class("face", 1, "face")
        for filelist in os.walk(dataset_dir):
            for file_dir in filelist[1]:
                print(file_dir)
                for subfilelist in os.walk(''.join([dataset_dir, "/", file_dir])):
                    for file in subfilelist[2]:
                        if file.endswith(".json"):
                            image_path = ''.join([dataset_dir, "/", file_dir, "/", file[:-4], "jpg"])
                            image = skimage.io.imread(image_path)
                            if image.shape[0] > 1024:
                                continue
                            # new_image = resize_image(image, DataSetConfig.IMAGE_MIN_DIM, DataSetConfig.IMAGE_MAX_DIM)
                            boundingboxes = json.load(open(''.join([dataset_dir, "/", file_dir, "/", file])))
                            self.add_image(
                                "face",
                                image_id=file[:-4]+"jpg",  # 使用文件名作为唯一id
                                path=image_path,
                                width=image.shape[1],
                                height=image.shape[0],
                                boundingbox=boundingboxes
                            )

    def load_mask(self, image_id):
        """
        Generate instance masks for shapes of given image ID
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "face":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['boundingbox'])], dtype=np.uint8)
        for i, p in enumerate(info['boundingbox'].values()):
            rr, cc = skimage.draw.polygon(p['y'], p['x'])
            mask[rr, cc, i] = 1
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """
    训练模型
    """
    print("start ready for train_set..")
    dataset_train = FaceDataSet()
    dataset_train.load_face("F:/wider_face_split/wider_face_train")
    dataset_train.prepare()
    print("start ready for val_set")
    dataset_val = FaceDataSet()
    dataset_val.load_face("F:/wider_face_split/wider_face_val")
    dataset_val.prepare()

    #model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect faces.')
    parser.add_argument("command", metavar="<command>", help="'train'")

    args = parser.parse_args()
    if args.command == "train":
        config = DataSetConfig()
    else:
        class InferenceConfig(DataSetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir='F:/model_dir')
        model.load_weights(model.find_last()[1], by_name=True)
        train(model)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir='F:/model_dir')
