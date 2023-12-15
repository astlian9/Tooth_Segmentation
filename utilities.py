import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile

clahe = cv2.createCLAHE( clipLimit=5, tileGridSize=(17, 17))

from glob import glob
import os

CHANNELS = 1

def parse_image(img_shape: list, img: str,class_dict):
    img_str = img
    img = tf.image.decode_png(tf.io.read_file(img),channels=CHANNELS)
    img = tf.image.resize(img,img_shape,antialias=True)
    img = tf.cast(img, tf.uint8)
    
    #base mask
    #inv class dict
    inv_class_dict = {value : key for key,value in class_dict.items()}
    msk = None
    for key in range(len(class_dict)):
      this_mask_layer = tf.strings.regex_replace(img_str,'Images',inv_class_dict[key])
      this_mask_layer = tf.image.decode_png(tf.io.read_file(this_mask_layer),channels=CHANNELS)
      this_mask_layer = tf.image.resize(this_mask_layer,img_shape,antialias=True)
      this_mask_layer = tf.where(this_mask_layer < 1, \
                                    np.dtype('uint8').type(0),
                                    np.dtype('uint8').type(1))
      this_mask_layer = tf.cast(this_mask_layer,tf.uint8)
      #add this mask layer to the mask
      #should check to see if there is any overlap
      if msk is None:
        msk = this_mask_layer
      else:
        msk = tf.concat([msk, this_mask_layer],axis=2)

    return (img, msk)

def augment_image(img,mask,enable):
  #from https://dergipark.org.tr/tr/download/article-file/1817118
  #horizontal flip resulted increased performance
  if enable[0]:
    do_flip = tf.random.uniform([]) > 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
  #image brightness
  if enable[1]:
    img = tf.image.adjust_brightness(img, \
            delta = tf.random.uniform([], minval=-0.10, maxval=0.10,\
                                     dtype=tf.dtypes.float32))
  
  #vertical flip
  if enable[2]:
    do_flip = tf.random.uniform([]) > 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_up_down(img), lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_up_down(mask), lambda: mask)
  #rotations
  if enable[3]:
    #get a random degree
    angle = tf.random.uniform([],minval=-20*np.pi/180,maxval=20*np.pi/180)
    img = tfa.image.rotate(images=img,angles=angle,fill_mode='constant',\
                            fill_value = 0)
    mask = tfa.image.rotate(images=mask,angles=angle,fill_mode='constant',\
                            fill_value = 0)
  #translations
  if enable[4]:
    dx = int(tf.random.uniform(
        [], minval=-int(0.2*img.shape[0]), maxval=+int(0.2*img.shape[0])))
    dy = int(tf.random.uniform([], minval=-int(0.2*mask.shape[1]),
             maxval=+int(0.2*mask.shape[1])))
    img = tfa.image.translate(images=img,translations = [dx,dy], fill_value = 0)
    mask = tfa.image.translate(images=mask,translations = [dx,dy], fill_value = 0)

  #salt n peppa
  if enable[5]:
    #lets figure out we can do 20% of all pixels
    random_values = tf.random.uniform(shape=img.shape)
    prob_salt = 0.05
    prob_pepper = 0.95
    img = tf.where(random_values < prob_salt, np.dtype('uint8').type(0), img)
    img = tf.where(random_values > prob_pepper, np.dtype('uint8').type(255), img)

  #image blurring, how does this work with the mask? should the mask be blurred as well?

  return (img,mask)

def get_class_dict(input_file):
  class_dict = {}
  with open(input_file,"r") as fh:
    lines = fh.readlines()
    ctr = 0
    for line in lines:
      if line.strip() not in class_dict:
        class_dict[line.strip()] = ctr
        ctr += 1
      else:
        #the line already exists, throw a warning
        warnings.warn("{} already added, skipping...".format(line))
  print(class_dict)
  return class_dict

def generate_dataset(dataPath, imageType, seed=42, img_shape=(512,512,1), batch_size = 16,\
                      enable_augmentation = (1,1,1,1,1,1), repeat_count=3, kfold = 1,
                      class_file = None):
    
    if class_file is None:
      raise ValueError("Class file undefined, please pass in")
    else:
      class_dict = get_class_dict(class_file)
    #list
    tf.random.set_seed(seed=seed) #set the seed.
    datasets = []
    img_shape = [img_shape[0], img_shape[1]]

    if len(enable_augmentation) != 6:
      raise ValueError("Enable augmentation value not of appropriate lengthrtr")

    if not any(enable_augmentation):
      print("no image augmentaton")
      repeat_count = 1
    else:
      print("image augmentation")
      

    files = tf.data.Dataset.list_files(os.path.join(dataPath,imageType), seed=seed).repeat(count=repeat_count)
    for k in range(kfold):
      train_dataset = files.shard(num_shards = kfold, index = k)
      train_dataset = train_dataset.shuffle(seed=seed, buffer_size = len(train_dataset), reshuffle_each_iteration=True)
      train_dataset = train_dataset.map(lambda x: parse_image(img_shape,x,class_dict),num_parallel_calls=tf.data.AUTOTUNE)
      train_dataset = train_dataset.map(lambda x,y: augment_image(x,y,enable_augmentation),\
                      num_parallel_calls=tf.data.AUTOTUNE)
      if batch_size is not None:
        train_dataset = train_dataset.batch(batch_size)
      train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
      datasets.append(train_dataset)

    return datasets,class_dict
    
def get_as_numpy(inputdata,with_clahe=0,with_unit_norm=0):
  fullX = []
  fullY = []
  ds = [x for x in tfds.as_numpy(inputdata)]
  for batch in ds:
    for x in range(len(batch[0])):
      fullX.append((batch[0][x]))
    for y in range(len(batch[1])):
      fullY.append(batch[1][y])

  return fullX,fullY

# GET TF DATASET AS A PATCHED IMAGE ARRAY
def get_patched(x,y):
  #x,y = get_as_numpy(inputdata)
  if len(x)>500: #can only take half of the x array...
    x=x[::3]
    y=y[::3]
  patchedImage = patchify_image(x[0])
  patchLen = len(patchedImage)
  p_sz_X = patchedImage[0].shape[0]
  p_sz_Y = patchedImage[1].shape[1]

  print((len(x)*patchLen, p_sz_X, p_sz_Y, x[0].shape[2]))
  patchedX = np.zeros(shape=(len(x)*patchLen, p_sz_X, p_sz_Y, x[0].shape[2]),dtype=patchedImage[0].dtype)
  patchedY = np.zeros(shape=(len(x)*patchLen, p_sz_X, p_sz_Y, y[0].shape[2]),dtype=patchedImage[0].dtype)
  for idx in range(len(x)):
    pX = patchify_image(x[idx])
    pX = pX.reshape(pX.shape + (1,))
    if y[idx].shape[-1]==1:
      pY = patchify_image(y[idx])
      pY = pY.reshape(pY.shape + (1,))
    else:
      pY = patchify_image(y[idx])

    patchedX[idx*patchLen:(idx+1)*patchLen] = pX
    patchedY[idx*patchLen:(idx+1)*patchLen] = pY
  print("Patching done")
  return patchedX, patchedY

#take given image and convert into 100 patches
def patchify_image(img2convert):
    data=[]
    img = cv2.resize(img2convert,(640,540))
    #patched into 100...patches
    size=100
    x_1,x_2=0,0
    y_1,y_2=0,0
    for i in range(0,10):
        #slide left to right
        y_1=size*i
        y_2=224+y_1
        if(y_2>=img.shape[1]):
            y_2=img.shape[1]
            y_1=y_2-224
            data.append(img[x_1:x_2,y_1:y_2])
            break
        for j in range(0,10):
            
            x_1=size*j
            x_2=224+x_1
            if(x_2>=img.shape[0]):
                x_2=img.shape[0]
                x_1=x_2-224
                data.append(img[x_1:x_2,y_1:y_2])
                break
            else:
                data.append(img[x_1:x_2,y_1:y_2])
    #dump in pickle
    data=np.array(data)
    return data

# this function pastes patch to its place
def patch_paste(big_image_piece,patch):
    for i in range(224):
        for j in range(224):
            if(big_image_piece[i][j]==-1):
                big_image_piece[i][j]=patch[i][j]
            else:
                big_image_piece[i][j]=(patch[i][j]+big_image_piece[i][j])/2
                
    return big_image_piece
    
# data will be of size (no_of_patches_of_each_image,224,224)
#this function is used to reconstruct the image from patches
def image_reconstruction(data, original_image_size=(512,512),size=100):
    img=np.zeros((original_image_size[1],original_image_size[0]))-1
    size=size
    x_1,x_2=0,0
    y_1,y_2=0,0
    patch_no=0
    for i in range(0,100):
        #slide left to right
        y_1=size*i
        y_2=224+y_1
        if(y_2>=img.shape[1]):
            y_2=img.shape[1]
            y_1=y_2-224
            img[x_1:x_2,y_1:y_2]= patch_paste(img[x_1:x_2,y_1:y_2] ,data[patch_no])
            patch_no=patch_no+1
            #print('x_1 :{}, x_2 :{}, y_1 :{}, y_2:{} '.format(x_1,x_2,y_1,y_2))
            break
        for j in range(0,100):
            x_1=size*j
            x_2=224+x_1
            if(x_2>=img.shape[0]):
                x_2=img.shape[0]
                x_1=x_2-224
                img[x_1:x_2,y_1:y_2]=patch_paste(img[x_1:x_2,y_1:y_2] ,data[patch_no])
                patch_no=patch_no+1
                #print('x_1 :{}, x_2 :{}, y_1 :{}, y_2:{} '.format(x_1,x_2,y_1,y_2))
                break
            else:
                img[x_1:x_2,y_1:y_2]= patch_paste(img[x_1:x_2,y_1:y_2] ,data[patch_no])
                patch_no=patch_no+1
                #print('x_1 :{}, x_2 :{}, y_1 :{}, y_2:{} '.format(x_1,x_2,y_1,y_2))
    return img*255
    #cv2_imshow(img*255)
    
def image_recon(patches, size=100):
# ONLY BECAUSE IT WAS RESIZED USING CV2 IN PATCHIFY_IMAGE()
    img=np.zeros((540,640))-1
    size=size
    w1, w2, h1, h2 = 0, 0, 0, 0
    size = size
    patch_num = 0

    for i in range(100):
        h1 = size * i
        h2 = h1 + 224
        if h2 >= img.shape[1]:
            h2 = img.shape[1]
            h1 = h2 - 224
            img[w1:w2,h1:h2] = patch_paste(img[w1:w2,h1:h2], patches[patch_num])
            patch_num += 1
            break

        elif patch_num <= len(patches):
            for j in range(100):
                w1 = size * j
                w2 = w1 + 224
                #print("h1:{}, h2:{}, w1:{}, w2:{}".format(h1,h2,w1,w2))
                if w2 >= img.shape[0]:
                    w2 = img.shape[0]
                    w1 = w2 - 224
                    img[w1:w2,h1:h2] = patch_paste(img[w1:w2,h1:h2], patches[patch_num])
                    patch_num += 1
                    break

                elif patch_num <= len(patches):
                    #print("here2")
                    img[w1:w2,h1:h2] = patch_paste(img[w1:w2,h1:h2], patches[patch_num])
                    patch_num += 1
                    
    return cv2.resize(img, (512,512))