#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import skimage.io
import skimage.draw
import json
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath('D:/Hanifan/Face-Detection-Base-on-Mask-R-CNN-master')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.utils import Dataset

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# In[2]:


class DatasetConfig(Config):
    GPU_COUNT = 1
    NAME = "FaceDetection"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGES_PER_GPU = 1 
    NUM_CLASSES = 1 + 1  # background + face
    STEPS_PER_EPOCH = 100


# In[3]:


class FaceDataset(Dataset):
    def load_face(self, dataset_dir):
        # Add classes
        countData = 0

        self.add_class("face", 1, "face")
        
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        images_dir = os.path.join(dataset_dir, 'images')
        
        for json_file in os.listdir(annotations_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(annotations_dir, json_file)
                
                with open(json_path, 'r') as jsonfile:
                    boundingboxes = json.load(jsonfile)
                
                image_filename = json_file[:-5] + ".jpg"
                
                image_path = None
                for subdir, _, files in os.walk(images_dir):
                    if image_filename in files:
                        image_path = os.path.join(subdir, image_filename)
                        break
                if image_path == None:
                    print(f"Image file does not exist: {image_filename}")
                    continue
                
                image = skimage.io.imread(image_path)
                if image.shape[0] > 1024:
                    continue
                
                self.add_image(
                    "face",
                    image_id=image_filename,  # Use filename as a unique id
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
        boundingboxes = info['boundingbox']

        # # Print bounding boxes for debugging
        # print(f"Bounding boxes for image {image_id}: {boundingboxes}")

        # Initialize mask
        mask = np.zeros([info['height'], info['width'], len(boundingboxes)], dtype=np.uint8)
        
        for i, key in enumerate(boundingboxes.keys()):
            box = boundingboxes[key]
            if 'x' in box and 'y' in box:
                x = np.clip(box['x'], 0, info['width'] - 1)
                y = np.clip(box['y'], 0, info['height'] - 1)
                rr, cc = skimage.draw.polygon(y, x)
                mask[rr, cc, i] = 1
            else:
                print(f"Invalid bounding box format for image {image_id}: {box}")
        
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "face":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)


# In[8]:


# Choose 'train' or 'inference'
if __name__ == "__main__":
    command = "train"

    if command == "train":
        config = DatasetConfig()
    else:
        class InferenceConfig(DatasetConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    if command == "train":
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        
        # Load pre-trained weights (optional, if available)
        coco_weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        if not os.path.exists(coco_weights_path):
            utils.download_trained_weights(coco_weights_path)

        # Load weights trained on MS-COCO
        model.load_weights(coco_weights_path, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
        # Training dataset
        print("Preparing training dataset...")
        dataset_train = FaceDataset()
        dataset_train.load_face("D:/Hanifan/Face-Detection-Base-on-Mask-R-CNN-master/wider_face_split/wider_face_train")
        dataset_train.prepare()

        # Validation dataset
        print("Preparing validation dataset...")
        dataset_val = FaceDataset()
        dataset_val.load_face("D:/Hanifan/Face-Detection-Base-on-Mask-R-CNN-master/wider_face_split/wider_face_val")
        dataset_val.prepare()

        # Proceed with training if all files are validated
        print("Starting training...")
        history = model.train(dataset_train, dataset_val,
                                learning_rate=config.LEARNING_RATE / 10,
                                epochs=30,
                                layers='heads')
        history = model.keras_model
        print(history)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)


# In[9]:

# model_path = os.path.join(MODEL_DIR, "mask_rcnn_face_new.h5")
# model.keras_model.save_weights(model_path)


# In[11]:


# from mrcnn.model import DataGenerator

# print("Preparing testing dataset...")
# dataset_test = FaceDataset()
# dataset_test.load_face("D:/Hanifan/Face-Detection-Base-on-Mask-R-CNN-master/wider_face_split/wider_face_test")
# dataset_test.prepare()


# def data_generator_for_prediction(dataset, config, shuffle=True, augment=False):
#     while True:
#         for image_id in dataset.image_ids:
#             image = dataset.load_image(image_id)
#             output = modellib.mold_image(image, config)
#             # Assuming the output contains more than 3 values, handle accordingly
#             molded_image, image_meta, windows, *_ = output
#             yield [molded_image, image_meta, windows]

# config = DatasetConfig()
# model_dir = os.path.join(ROOT_DIR, "logs")
# # Instantiate generator
# val_generator = data_generator_for_prediction(dataset_test, config)

# # Custom function to filter inputs for prediction
# def filter_prediction_inputs(data):
#     # Assuming data is a tuple of (images, additional_inputs)
#     images = data[0]
#     return images

# # Modified prediction call
# model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
# predictions = []
# for batch in val_generator:
#     filtered_batch = filter_prediction_inputs(batch)
#     batch_predictions = model.keras_model.predict(filtered_batch)
#     predictions.extend(batch_predictions)


# # Initialize the Data Generator
# # val_generator = DataGenerator(dataset_test, config, model_dir, shuffle=True)
# val_steps = len(dataset_test.image_ids) // config.BATCH_SIZE

# # Predict using the model directly
# model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
# predictions = model.keras_model.predict(val_generator, steps=val_steps)
# print(predictions)


# # In[ ]:


# # Check model's input
# print(model.keras_model.input)


# In[24]:


# import tensorflow as tf
# import numpy as np

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.dense = tf.keras.layers.Dense(100, kernel_initializer="ones", trainable=False)
#         self.dropout = tf.keras.layers.Dropout(0.5)

#     def call(self, inputs, training=False):
#         x = self.dense(inputs)
#         if training:
#             x = self.dropout(x, training=training)
#         x = tf.reshape(tf.reduce_sum(x) / 100., [1, 1])  # Reshape to prevent aggregation error
#         return x

# model = MyModel()

# def loss(y_true, y_pred):
#     return y_pred

# model.compile(optimizer="sgd", loss=loss)

# x = np.ones((1, 1), dtype=np.float32)

# print("Predicting:")
# print(model.predict(x))  # No dropout, output is 1 as expected

# print("Training:")
# print(model.train_on_batch(x, np.zeros((1, 1), dtype=np.float32)))  # Train the model

