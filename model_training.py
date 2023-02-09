import tensorflow as tf
from bnn_model import BNNModel


def main():
    bnn_model_obj = BNNModel(n_inputs=1280,n_outputs=200)
    bnn_model = bnn_model_obj.create_bnn_model(train_size=5000,activation='selu')
    
    print(bnn_model.summary())
    
    
    
    
    
    
if __name__=="__main__":
    main()