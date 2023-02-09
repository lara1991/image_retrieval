import tensorflow as tf
import tensorflow_probability as tfp

class BNNModel:
    def __init__(self,n_inputs,n_outputs,hidden_units=[2,2]):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_units = hidden_units
        self.model = None
        
    def prior(self,kernel_size,bias_size,dtype=None):
        n = kernel_size + bias_size
        prior_model = tf.keras.Sequential([
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n),scale_diag=tf.ones(n)
                )
            )
        ])
        
        return prior_model
    
    def posterior(self,kernel_size,bias_size,dtype=None):
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential([
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n),dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ])
        
        return posterior_model
    
    def create_bnn_model(self,train_size,activation='selu'):
        inputs = tf.keras.layers.Input(shape=(self.n_inputs,))
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = inputs
        
        for hidden_unit in self.hidden_units:
            x = tfp.layers.DenseVariational(
                units=hidden_unit,
                make_prior_fn=self.prior,
                make_posterior_fn=self.posterior,
                kl_weight=1 / train_size,
                activation=activation
            )(x)
        
        x = tfp.layers.DenseVariational(
            units=self.n_outputs,
            make_prior_fn=self.prior,
            make_posterior_fn=self.posterior,
            kl_weight=1 / train_size,
            activation=activation
        )(x)
        
        outs = tfp.layers.OneHotCategorical(self.n_outputs,convert_to_tensor_fn=tfp.distributions.Distribution.mode)(x)
        
        self.model = tf.keras.models.Model(inputs=inputs,outputs=outs,name="bnn_classification_model")
        return self.model
    
    def nll_loss(self,y_true,y_pred):
        return -y_pred.log_prob(y_true)
        
        