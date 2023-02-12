import pandas as pd
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier


from paths_links import FEATURES_DIR

def load_data_features(feature_data_file_name="train_features.csv"):
    train_features_path = FEATURES_DIR + "/" + feature_data_file_name
    # col_names = ['class_label','feature_vec']
    df = pd.read_csv(train_features_path,header=None)
    # print(df.head())
    return df

def fit_knn_model(x,y,num_neighbours=5):
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbours,weights='distance',)
    knn_model.fit(X=x,y=y)
    return knn_model

def main():
    train_dataset = load_data_features(feature_data_file_name="train_features.csv")
    vals = train_dataset.values
    y_labels,x_features = vals[:,0],vals[:,1:]
    
    test_dataset = load_data_features(feature_data_file_name="test_features.csv")
    test_vals = test_dataset.values[:2000]
    x_test,y_test = test_vals[:,1:],test_vals[:,0]
          
    # print("Fitting the kNN model")
    # for k in range(1,250):
    #     trained_knn_model = fit_knn_model(x=x_features,y=y_labels,num_neighbours=k)    
    #     ms = trained_knn_model.score(X=x_test,y=y_test)
    #     print(f"k val: {k} Mean accuracy: {ms}")
    
    trained_knn_model = fit_knn_model(x=x_features,y=y_labels,num_neighbours=8)    
    ms = trained_knn_model.score(X=x_test,y=y_test)
    print(f"k val: {k} Mean accuracy: {ms}")
    
    
    
if __name__=="__main__":
    main()
    