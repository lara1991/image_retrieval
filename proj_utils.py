import tensorflow as tf
import numpy as np
import pandas as pd 
import cv2
import os
import tqdm

from paths_links import *

def load_datafiles():
    image_paths = []
    image_labels = []
    train_test_split = []
    
    with open(IMAGE_PATHS,'r') as file:
        data_img_paths = file.readlines()
        # data_img_paths = data_img_paths.split("\n")
    with open(IMAGE_CLASS_LABELS,'r') as file:
        data_img_class_labels = file.readlines()
        # data_img_class_labels = data_img_class_labels.split('\n')
    with open(TRAIN_TEST_SPLIT,'r') as file:
        data_train_test_split = file.readlines()
        # data_train_test_split = data_train_test_split.split('\n')
        
    for vals in zip(data_img_paths,data_img_class_labels,data_train_test_split):
        
        img_path = vals[0].split(" ")[-1][:-1]
        img_label = vals[1].split(" ")[-1][:-1]
        train_test =  vals[2].split(" ")[-1][:-1]
        
        image_paths.append(img_path)
        image_labels.append(img_label)
        train_test_split.append(train_test)
    
    df = pd.DataFrame(
        {
            'img_paths' : image_paths,
            'img_labels': image_labels,
            'train_test': train_test_split
        }
    )
    
    return df

def get_feature_extraction_model(img_size = (160,160,3),save_feature_extractor="feature_extractor_model.h5"):
    feature_extractor_base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top = False,
        weights='imagenet'
    )
    
    features = feature_extractor_base_model.output
    
    avg_outs = tf.keras.layers.GlobalAveragePooling2D()(features)
    feature_extractor_model = tf.keras.models.Model(
        inputs=feature_extractor_base_model.inputs,
        outputs=avg_outs
    )
    
    feature_extractor_base_model.trainable = False
    
    if save_feature_extractor:
        model_save_path = os.path.join(FEATURES_DIR,save_feature_extractor)
        feature_extractor_model.save(model_save_path)
    
    return feature_extractor_model

def input_datapipeline(df):
    pass