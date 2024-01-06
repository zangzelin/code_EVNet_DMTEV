
#%%
import numpy as np
import tensorflow as tf
import random as rn
import os

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

K.set_session(sess)
#----------------------------Reproducible----------------------------------------------------------------------------------------

key_feture_number=50
epochs_number=1000
batch_size_value=128
is_use_bias=True
num_data_used=10000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#%%
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Activation, Dropout, Layer
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import optimizers,initializers,constraints,regularizers
from keras import backend as K
from keras.callbacks import LambdaCallback,ModelCheckpoint
from keras.utils import plot_model

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
# matplotlib.style.use('ggplot')

import random
import scipy.sparse as sparse

#--------------------------------------------------------------------------------------------------------------------------------
#Import defined methods
import sys
sys.path.append(r"./")
import baseline.FAE.Defined.Functions as F

#%%

# np.random.seed(seed)
# (x_train_, y_train_), (x_test_, y_test_) = mnist.load_data()
# x_data=np.r_[x_train_,x_test_].reshape(70000, 28*28).astype('float32')/255.0
# y_data=np.r_[y_train_,y_test_]

# np.random.seed(seed)
# x_data_num,_=x_data.shape
# index=np.arange(x_data_num)
# np.random.shuffle(index)

# data_arr=x_data[index][0:num_data_used]
# label_arr_onehot=y_data[index][0:num_data_used]

# C_train_x,C_test_x,C_train_y,C_test_y= train_test_split(data_arr,label_arr_onehot,test_size=0.2,random_state=seed)
# x_train,x_validate,y_train_onehot,y_validate_onehot= train_test_split(C_train_x,C_train_y,test_size=0.1,random_state=seed)
# x_test=C_test_x
# y_test_onehot=C_test_y

# print('Shape of x_train: ' + str(x_train.shape)) 
# print('Shape of x_validate: ' + str(x_validate.shape)) 
# print('Shape of x_test: ' + str(x_test.shape))
# print('Shape of y_train: ' + str(y_train_onehot.shape))
# print('Shape of y_validate: ' + str(y_validate_onehot.shape))
# print('Shape of y_test: ' + str(y_test_onehot.shape))

# print('Shape of C_train_x: ' + str(C_train_x.shape)) 
# print('Shape of C_train_y: ' + str(C_train_y.shape)) 
# print('Shape of C_test_x: ' + str(C_test_x.shape)) 
# print('Shape of C_test_y: ' + str(C_test_y.shape)) 

# F.show_data_figures(x_train_[0:120],28,28,40)



#--------------------------------------------------------------------------------------------------------------------------------
class Feature_Select_Layer(Layer):
    
    def __init__(self, output_dim, l1_lambda, **kwargs):
        super(Feature_Select_Layer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.l1_lambda=l1_lambda

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',  
                                      shape=(input_shape[1],),
                                      initializer=initializers.RandomUniform(minval=0.999999, maxval=0.9999999, seed=seed),
                                      trainable=True,
                                      regularizer=regularizers.l1(self.l1_lambda),
                                      constraint=constraints.NonNeg())
        super(Feature_Select_Layer, self).build(input_shape)
    
    def call(self, x, selection=False,k=key_feture_number):
        kernel=self.kernel        
        if selection:
            kernel_=K.transpose(kernel)
            kth_largest = tf.math.top_k(kernel_, k=k)[0][-1]
            kernel = tf.where(condition=K.less(kernel,kth_largest),x=K.zeros_like(kernel),y=kernel)        
        return K.dot(x, tf.linalg.tensor_diag(kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#--------------------------------------------------------------------------------------------------------------------------------
def Fractal_Autoencoder(p_data_feature,\
                        p_feture_number,\
                        p_encoding_dim,\
                        p_learning_rate=1E-3,\
                        p_l1_lambda=0.1,\
                        p_loss_weight_1=1,\
                        p_loss_weight_2=2,\
                        p_is_use_bias=True):
    
    input_img = Input(shape=(p_data_feature,), name='autoencoder_input')

    feature_selection = Feature_Select_Layer(output_dim=p_data_feature,\
                                             l1_lambda=p_l1_lambda,\
                                             input_shape=(p_data_feature,),\
                                             name='feature_selection')

    feature_selection_score=feature_selection(input_img)
    feature_selection_choose=feature_selection(input_img,selection=True,k=p_feture_number)

    encoded = Dense(p_encoding_dim,\
                    activation='linear',\
                    kernel_initializer=initializers.glorot_uniform(seed),\
                    use_bias=p_is_use_bias,\
                    name='autoencoder_hidden_layer')
    
    encoded_score=encoded(feature_selection_score)
    encoded_choose=encoded(feature_selection_choose)
    
    bottleneck_score=encoded_score
    bottleneck_choose=encoded_choose
    
    decoded = Dense(p_data_feature,\
                    activation='linear',\
                    kernel_initializer=initializers.glorot_uniform(seed),\
                    use_bias=p_is_use_bias,\
                    name='autoencoder_output')
    
    decoded_score =decoded(bottleneck_score)
    decoded_choose =decoded(bottleneck_choose)

    latent_encoder_score = Model(input_img, bottleneck_score)
    latent_encoder_choose = Model(input_img, bottleneck_choose)
    feature_selection_output=Model(input_img,feature_selection_choose)
    autoencoder = Model(input_img, [decoded_score,decoded_choose])
    
    autoencoder.compile(loss=['mean_squared_error','mean_squared_error'],\
                        loss_weights=[p_loss_weight_1, p_loss_weight_2],\
                        optimizer=optimizers.Adam(lr=p_learning_rate))
    
    print('Autoencoder Structure-------------------------------------')
    autoencoder.summary()
    return autoencoder,feature_selection_output,latent_encoder_score,latent_encoder_choose

def FAEFS(x_train, x_validate, x_test, key_feture_number, epochs_number=500, batch_size_value=128):

    F_AE,\
    feature_selection_output,\
    latent_encoder_score_F_AE,\
    latent_encoder_choose_F_AE=Fractal_Autoencoder(p_data_feature=x_train.shape[1],\
                                                p_feture_number=key_feture_number,\
                                                p_encoding_dim=key_feture_number,\
                                                p_learning_rate= 1E-3,\
                                                p_l1_lambda=0.1,\
                                                p_loss_weight_1=1,\
                                                p_loss_weight_2=2,\
                                                p_is_use_bias=is_use_bias)

    F_AE_history = F_AE.fit(x_train, [x_train,x_train],\
                            epochs=epochs_number,\
                            batch_size=batch_size_value,\
                            shuffle=True,\
                            validation_data=(x_validate, [x_validate,x_validate]),\
                            # callbacks=[model_checkpoint]
                            )

    # p_data=F_AE.predict(x_test)
    f = feature_selection_output.predict(x_test).mean(axis=0)
    return f
# %%

if __name__ == '__main__':

    seed = 1
    np.random.seed(seed)
    (x_train_, y_train_), (x_test_, y_test_) = mnist.load_data()
    x_data=np.r_[x_train_,x_test_].reshape(70000, 28*28).astype('float32')/255.0
    y_data=np.r_[y_train_,y_test_]

    np.random.seed(seed)
    x_data_num,_=x_data.shape
    index=np.arange(x_data_num)
    np.random.shuffle(index)

    data_arr=x_data[index][0:num_data_used]
    label_arr_onehot=y_data[index][0:num_data_used]

    C_train_x,C_test_x,C_train_y,C_test_y= train_test_split(data_arr,label_arr_onehot,test_size=0.2,random_state=seed)
    x_train,x_validate,y_train_onehot,y_validate_onehot= train_test_split(C_train_x,C_train_y,test_size=0.1,random_state=seed)
    x_test=C_test_x
    y_test_onehot=C_test_y

    print('Shape of x_train: ' + str(x_train.shape)) 
    print('Shape of x_validate: ' + str(x_validate.shape)) 
    print('Shape of x_test: ' + str(x_test.shape))
    print('Shape of y_train: ' + str(y_train_onehot.shape))
    print('Shape of y_validate: ' + str(y_validate_onehot.shape))
    print('Shape of y_test: ' + str(y_test_onehot.shape))


    key_feture_number=50
    epochs_number=10
    batch_size_value=128
    is_use_bias=True
    num_data_used=10000

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    o,f = FAEFS(x_train, x_validate, x_test, key_feture_number, epochs_number)
    print(f)