import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from bnn_model import BNNModel
from proj_utils import input_datapipeline,load_datafiles,get_feature_extraction_model



def extract_feature_vectors(image,label):
    input_shape = image.shape[1:]
    print(input_shape)

    feature_extraction_model = get_feature_extraction_model(input_shape)
    extracted_features = feature_extraction_model(image)
    # feature_vec = ",".join([str(v) for v in extracted_features])

    # print(feature_vec)
    # feature_vec = np.array(feature_vec)
    return (extracted_features,label)

def main():
    bnn_model_obj = BNNModel(n_inputs=1280,n_outputs=200)

    df = load_datafiles()

    train_df = df[df['train_test'] == '1']
    train_df = train_df.sample(frac=1)
    train_df,val_df = train_test_split(train_df,test_size=0.2) 


    train_ds = input_datapipeline(train_df)
    train_ds = train_ds.map(extract_feature_vectors,num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = input_datapipeline(val_df,train=False)
    val_ds = val_ds.map(extract_feature_vectors,num_parallel_calls=tf.data.AUTOTUNE)

    # for train_d in train_ds.take(1):
    #     print(train_d)

    ## bnn model
    number_of_train_datasamples = train_df.shape[0]
    bnn_model = bnn_model_obj.create_bnn_model(train_size=number_of_train_datasamples,activation='selu')
    print(bnn_model.summary())

    ## compile model
    optim = tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.9
    )

    bnn_model.compile(loss=bnn_model_obj.nll_loss,optimizer=optim,metrics=['accuracy'])

    H = bnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2
    )
    

    
    
    
    
    
    
if __name__=="__main__":
    main()