from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,Conv1D,Cropping2D,Concatenate,Dropout,BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras.models import load_model
from keras.initializers import RandomNormal
import warnings

def UNET(input_shape=(512,512,1),num_classes = 1):
  if input_shape != (512,512,1):
    raise ValueError("This image size is unsupported, change size or update code.")

  input = Input(shape = input_shape)
  #init = RandomNormal(stddev = 0.2)
  init = 'he_normal'

  #Model consists of 4 contraction, and then 4 expansions
  #each expansion concats the downward expansion
  
  #helper funcs to build model
  def block_layer(input,filters,kernel_initializer, \
                        activation="relu",name=None, doBatchNorm = True):
    conv1 = Conv2D(filters,(3,3), activation = activation, padding = 'same',\
                   kernel_initializer = kernel_initializer, name = "{}A".format(name))(input)
    d1=Dropout(0.1, name = "{}B".format(name))(conv1)
    conv2 = Conv2D(filters,(3,3), activation = activation, padding = 'same',\
                   kernel_initializer = kernel_initializer, name = "{}C".format(name))(d1)
    if doBatchNorm:
      b=BatchNormalization(name = "{}D".format(name))(conv2)
      return b
    else: 
      return conv2

  def expand_and_concat(inputBefore,inputSame,filters,kernel_initializer,\
                        activation="relu",name=None):
    conv = Conv2DTranspose(filters,(4,4), activation = activation, name = "{}up".format(name),\
                             padding = 'same', strides=(2,2),kernel_initializer = kernel_initializer)(inputBefore)
    x = Concatenate(name = "{}combine".format(name))([conv,inputSame])
    return x
  #~~~
  #build model
  #512->256
  down1 = block_layer(input=input,filters=32,kernel_initializer=init,name="Down1")
  #256->128
  down2 = block_layer(input=MaxPooling2D(pool_size=(2,2))(down1),filters=64,kernel_initializer=init,name="Down2")
  #128->64
  down3 = block_layer(input=MaxPooling2D(pool_size=(2,2))(down2),filters=128,kernel_initializer=init,name="Down3")
  #64->32
  down4 = block_layer(input=MaxPooling2D(pool_size=(2,2))(down3),filters=256,kernel_initializer=init,name="Down4")
  #BEG BASE~~~
  base = block_layer(input=MaxPooling2D(pool_size=(2,2))(down4),filters=512,kernel_initializer=init,name="Base")
  #END BASE~~~
  #32->64
  up4 = block_layer(input = expand_and_concat(base,down4,filters=512,kernel_initializer=init,name="Up4"),\
                    filters=256,kernel_initializer=init,name="Up4")
  #64->128
  up3 = block_layer(input = expand_and_concat(up4,down3,filters=256,kernel_initializer=init,name="Up3"),\
                    filters=128,kernel_initializer=init,name="Up3")
  #128->256
  up2 = block_layer(input = expand_and_concat(up3,down2,filters=128,kernel_initializer=init,name="Up2"),\
                    filters=64,kernel_initializer=init,name="Up2")
  #256->512
  up1 = block_layer(input = expand_and_concat(up2,down1,filters=64,kernel_initializer=init,name = "Up1"),\
                    filters=32,kernel_initializer=init,name="Up1", doBatchNorm=False)
  #~~~

  #and convolve base image with mask.
  print('sigmoid' if num_classes==1 else 'sigmoid')
  out = Conv2D(num_classes,(1,1), activation = 'sigmoid' if num_classes==1 else 'sigmoid' ,\
               padding = 'same', kernel_initializer = init,name = "Final")(up1)
  
  #model should now just be
  return Model(input,out)