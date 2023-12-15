
import tensorflow as tf
from tensorflow import keras
from keras import models,layers
from keras.layers import Input


def FCN_encoder(x):
    x=layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x) 
    x=layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    p1=layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)

    x=layers.Conv2D(128, (3, 3), padding="same", activation="relu")(p1)
    x=layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    p2=layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)

    x=layers.Conv2D(256, (3, 3), padding="same", activation="relu")(p2)
    x=layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x=layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    p3=layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)

    x=layers.Conv2D(512, (3, 3), padding="same", activation="relu")(p3)
    x=layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x=layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    p4=layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)


    x=layers.Conv2D(1024, (3, 3), padding="same", activation="relu")(p4)
    x=layers.Conv2D(1024, (3, 3), padding="same", activation="relu")(x)
    x=layers.Conv2D(1024, (3, 3), padding="same", activation="relu")(x)
    p5=layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)

    return p1,p2,p3,p4,p5

def FCN(x,n_classes):
    op=layers.Conv2D(n_classes,(1,1),strides=1,activation="relu",kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                kernel_regularizer = tf.keras.regularizers.L2(1e-3))(x)
    return op

def FCN_decoder(p1,p2,p3,p4,p5,n_classes=2):

    encoder_out= FCN(p5,n_classes)
    # num_classes. num_classes should be 2 for this project -- a
    # Transposed/backward convolutions for creating a decoder
    deconv_1 = layers.Conv2DTranspose( n_classes, 4, 2, 'SAME', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01) ,
                                      kernel_regularizer = keras.regularizers.l2(1e-3))(encoder_out)
    
    # Add a skip connection to previous VGG layer

    skip_1= FCN(p4,n_classes)
    add1=layers.Add()([deconv_1,skip_1])

    # Up-sampling
    deconv_2 = layers.Conv2DTranspose( n_classes, 4, 2, 'SAME', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                                kernel_regularizer = keras.regularizers.L2(1e-3))(add1)

    skip_2 = FCN(p3,n_classes)
    add2=layers.Add()([deconv_2,skip_2])

    deconv_3 = layers.Conv2DTranspose(n_classes, 16, 8, 'SAME', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                                kernel_regularizer = keras.regularizers.L2(1e-3))(add2 )
    
    return deconv_3

def PatchFCN(input_shape=(512,512,1),num_classes = 1):
    input=Input(shape=input_shape)
    p1,p2,p3,p4,p5=FCN_encoder(input) # encoder
    output=FCN_decoder(p1,p2,p3,p4,p5,n_classes=num_classes) # decoder
    final_output=layers.Activation('sigmoid')(output) #doing this to enable multi-label classification

    model=models.Model(input,final_output)

    return model
