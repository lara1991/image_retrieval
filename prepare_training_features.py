
import tensorflow as tf
import numpy as np
import cv2
import os
import tqdm

from paths_links import *
from proj_utils import load_datafiles,get_feature_extraction_model


def extracting_features(dataframe,feature_extractor,feature_set_file_name="train_features.csv",img_size=(160,160),train_test = "1"):
    
    feature_set_file_path = os.path.join(FEATURES_DIR,feature_set_file_name)
    
    features_save_file = open(feature_set_file_path,'w')
    
    df_training_set = dataframe[dataframe['train_test'] == train_test]
    
    # print(dataframe['train_test'])
    print(df_training_set.head())
    
    for idx,row in tqdm.tqdm(df_training_set.iterrows(),total=df_training_set.shape[0]):
        img_path = os.path.join(IMAGE_DIR,row[0])
        
        # print(img_path)
        label = row[1]
        
        img = cv2.imread(img_path)
        # print(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,img_size)
        img = np.array(img).astype(np.float32) / 255.
        img = np.expand_dims(img,0)
        
        extracted_features = feature_extractor.predict(img)
        feature_vec = ",".join([str(v) for v in extracted_features])
        features_save_file.write("{},{}\n".format(label,feature_vec))
    
    features_save_file.close()
  
def main():
    dataframe = load_datafiles()
    # print(dataframe['img_paths'].head())
    # print(dataframe['img_labels'].tail())
    # print(dataframe['train_test'].tail())
    
    feature_extractor = get_feature_extraction_model()
    # print(feature_extractor.summary())
    
    extracting_features(
        dataframe,
        feature_extractor
    )

if __name__ == '__main__':
    main()
    




