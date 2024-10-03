'''This file reads a trained gaze prediction network by Zhang et al. 2020, and a data file, then outputs human attention map
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).'''

import tensorflow as tf, numpy as np, tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

from atari_cr.atari_head.og_dataset import *

def my_softmax(x):
    """Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    print("\nx: ", x)
    original_shape = tf.shape(x)
    x = tf.reshape(x, [original_shape[0], -1])
    x = tf.nn.softmax(x, axis=1)
    x = tf.reshape(x, original_shape)
    return x

def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    epsilon = 1e-10 # introduce epsilon to avoid log and division by zero error
    y_true = K.backend.clip(y_true, epsilon, 1)
    y_pred = K.backend.clip(y_pred, epsilon, 1)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = [1,2,3])

class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 

    def init_model(self, gaze_model_file):
        # Constants
        self.k = 4
        self.stride = 1
        self.img_shape = 84

        # Constants
        SHAPE = (self.img_shape,self.img_shape,self.k) # height * width * channel
        dropout = 0.0
        ###############################
        # Architecture of the network #
        ###############################
        inputs=L.Input(shape=SHAPE)
        x=inputs 
        
        conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
        x = conv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv2=L.Conv2D(64, (4,4), strides=2, padding='valid')
        x = conv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv3=L.Conv2D(64, (3,3), strides=1, padding='valid')
        x = conv3(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
        x = deconv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
    
        deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
        x = deconv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)         
    
        deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
        x = deconv3(x)
    
        outputs = L.Activation(my_softmax)(x)
        print("outputs: ", outputs)
        self.model=Model(inputs=inputs, outputs=outputs)
        opt=K.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=my_kld, optimizer=opt)
        
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
  
    def predict(self, imgs):
        print("Predicting results...")
        self.preds = self.model.predict(imgs) 
        print("Predicted.")
        return self.preds

    def predict_and_save(self, imgs):
        print("Predicting results...")
        self.preds = self.model.predict(imgs) 
        print("Predicted.")
    
        print("Writing predicted gaze heatmap (train) into the npz file...")
        np.savez_compressed("human_gaze_" + self.game_name, heatmap=self.preds[:,:,:,0])
        print("Done. Output is:")
        print(" %s" % "human_gaze_" + self.game_name + '.npz')

if __name__ == "__main__":
    game_name = sys.argv[3]

    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    # For gaze prediction
    d.generate_data_for_gaze_prediction()
    
    gp = Human_Gaze_Predictor(game_name) #game name
    gp.init_model(sys.argv[4]) #gaze model .hdf5 file provided in the repo

    # tf.saved_model.save(gp.model, "saved_model")
    preds = gp.predict(d.gaze_imgs)
    preds = tf.reshape(preds, preds.shape[:-1])
    with open(f"output/og_{game_name}_{sys.argv[1].split('/')[-1].split('.')[0]}_preds.np", "wb") as f: np.save(f, preds)
    breakpoint()

