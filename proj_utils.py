import tensorflow as tf
import numpy as np
import pandas as pd 
import os

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

    df['img_labels'] = df['img_labels'].astype(np.int32)
    
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
    
    # if save_feature_extractor:
    #     model_save_path = os.path.join(FEATURES_DIR,save_feature_extractor)
    #     feature_extractor_model.save(model_save_path)
    
    return feature_extractor_model


def get_img_and_label(tensor_val):
    img_path = IMAGE_DIR + "/" + tensor_val['img_paths']
    img_label = tensor_val['img_labels']
    img_label = tf.one_hot(img_label,depth=200)

    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img,channels=3)
    img = tf.image.convert_image_dtype(img,dtype=tf.float32) / 255.
    img.set_shape([160,160,3])
    img = tf.image.resize(img,[160,160])
    # print("min max: {} {}".format(tf.reduce_min(img),tf.reduce_max(img)))

    # img_label = tf.cast(img_label,dtype=tf.int32)

    return (img,img_label)

def image_data_augmentation(image,label):
    # images = images / 255. ## rescaling images

    # print(image)

    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    
    return (image,label)


def input_datapipeline(df,batch_size=4,train=True):
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    if train:
        dataset = dataset.shuffle(df.shape[0],seed=42)
    dataset = dataset.map(get_img_and_label,num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    if train:
        dataset = dataset.map(image_data_augmentation,num_parallel_calls=tf.data.AUTOTUNE,)
    # dataset = dataset.map(extract_feature_vectors,num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)


    # print("\nInside the pipeline")
    # for data in dataset.take(2):
    #     print(data)
    return dataset




## testing area - main functions

def main_datapipeline():
    pass


if __name__=="__main__":
    main_datapipeline()