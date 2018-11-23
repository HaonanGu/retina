# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:29:03 2018

@author: ghn
"""
# -*- coding: utf-8 -*-
import numpy as np
from keras import layers
from keras.regularizers import l2
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K

import numpy as np
import keras
import configparser
import sys
sys.path.append('./lib')
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,normalization,concatenate,Activation
from keras.optimizers import Adam,Adagrad
from keras.metrics import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
#from keras.regularizers import l2,l1, activity_l2
import random
import sys
#sys.path.append('/home/customer/document/retina-unet-tuned/')
from help_functions import *
from extract_patches import get_data_training
from pre_processing import *
import matplotlib.pyplot as plt
#function to obtain data for training/testing (validation)
K.set_image_dim_ordering('th')
NewTrain=True
ChangeReg=False
config = configparser.RawConfigParser()
#config.read('./configuration.txt')
config.read('./configuration.txt')
Tensorflow=int(config.get('data attributes', 'Tensorflow'))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def get_unet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch,patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(inputs)#'valid'
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), W_regularizer=l2(0.01),padding='same')(pool1) #,activation='relu', padding='same')(pool1)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 =  Conv2D(64, (3, 3),W_regularizer=l2(0.01), padding='same')(conv2)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv2)
    shortcut2 = Conv2D(64, (1, 1),W_regularizer=l2(0.01), padding='same')(pool1)
    conv2 = Add()([conv2,shortcut2])
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), W_regularizer=l2(0.01),padding='same')(pool2)   #, activation='relu', padding='same')(pool2)
    conv3 = normalization.BatchNormalization(epsilon=2e-05,axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv3)
    conv3 = Activation('relu')(conv3)
    #conv3 = Dropout(0.3)(conv3)
    conv3 =  Conv2D(128, (3, 3), W_regularizer=l2(0.01),padding='same')(conv3)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv3)
    shortcut3 = Conv2D(128, (1, 1), W_regularizer=l2(0.01),padding='same')(pool2)
    conv3 = Add()([conv3,shortcut3])
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3),W_regularizer=l2(0.01), padding='same')(pool3)   #, activation='relu', padding='same')(pool2)
    conv4 = normalization.BatchNormalization(epsilon=2e-05,axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 =  Conv2D(256, (3, 3), W_regularizer=l2(0.01),padding='same')(conv4)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv4 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv4)
    shortcut4 = Conv2D(256, (1, 1), W_regularizer=l2(0.01),padding='same')(pool3)
    conv4 = Add()([conv4,shortcut4])
    conv4 = Activation('relu')(conv4 )
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(pool4)   #, activation='relu', padding='same')(pool2)
    conv5 = normalization.BatchNormalization(epsilon=2e-05,axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(conv5)
    #conv5 = Dropout(0.3)(conv5)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(up1)
    conv6 = Dropout(0.3)(conv6)
    #conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    conv6 =  Conv2D(256, (3, 3),W_regularizer=l2(0.01), padding='same')(conv6)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    shortcut6 = Conv2D(256, (1, 1), W_regularizer=l2(0.01),padding='same')(up1)
    conv6 = Add()([conv6,shortcut6])
    conv6 = Activation('relu')(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(up2)
    conv7 = Dropout(0.3)(conv7)
    #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 =  Conv2D(128, (3, 3),W_regularizer=l2(0.01), padding='same')(conv7)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv7 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv7)
    shortcut7 = Conv2D(128, (1, 1),W_regularizer=l2(0.01), padding='same')(up2)
    conv7 = Add()([conv7,shortcut7])
    conv7 = Activation('relu')(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu',W_regularizer=l2(0.01), padding='same')(up3)
    conv8 = Dropout(0.3)(conv8)
    conv8 =  Conv2D(64, (3, 3),W_regularizer=l2(0.01), padding='same')(conv8)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv8 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv8)
    shortcut8 = Conv2D(64, (1, 1), W_regularizer=l2(0.01),padding='same')(up3)
    conv8 = Add()([conv8,shortcut8])
    conv8 = Activation('relu')(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', W_regularizer=l2(0.01),padding='same')(up4)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', W_regularizer=l2(0.01),padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='relu',W_regularizer=l2(0.01),padding='same')(conv9)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)

    act = Activation('softmax')(conv10)

    model = Model(inputs=inputs, outputs=act)
    return model

def CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,Nimgs):
    class_weight=class_weight/np.sum(class_weight)
    p = random.uniform(0,1)
    psum=0
    label=0
    for i in range(class_weight.shape[0]):
        psum=psum+class_weight[i]
        if p<psum:
            label=i
            break
    if label==class_weight.shape[0]-1:
        i_center = random.randint(0,Nimgs-1)
        x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
         # print "x_center " +str(x_center)
        y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
    else:
        t=mlist[label]
        cid=random.randint(0,t[0].shape[0]-1)
        i_center=t[0][cid]
        y_center=t[1][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
        x_center=t[2][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
        #mask_shape=train_masks.shape[3]

        if y_center<patch_w/2:
            y_center=patch_w/2
        elif y_center>img_h-patch_w/2:
            y_center=img_h-patch_w/2

        if x_center<patch_w/2:
            x_center=patch_w/2
        elif x_center>img_w-patch_w/2:
            x_center=img_w-patch_w/2

    return i_center,x_center,y_center



def Active_Generate(train_imgs,train_masks,patch_h,patch_w,batch_size,N_subimgs,N_imgs,class_weight,mlist):
    while 1:
        img_h=train_imgs.shape[2]
        img_w=train_imgs.shape[3]
        for t in range(int(N_subimgs*N_imgs/batch_size)):
            X=np.zeros([batch_size,3,patch_h,patch_w])
            Y=np.zeros([batch_size,patch_h*patch_w,num_lesion_class+1])
            for j in range(batch_size):
                [i_center,x_center,y_center]=CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,N_imgs)
                patch = train_imgs[i_center,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                patch_mask = train_masks[i_center,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                X[j,:,:,:]=patch
                Y[j,:,:]=masks_Unet(np.reshape(patch_mask,[1,1,patch_h,patch_w]),1)
            yield (X, Y)

def SampleTest(gen,batch_size,patch_h,patch_w):
    (X,Y)=next(gen)
    for i in range(batch_size):
        plt.imshow(X.transpose(0,2,3,1)[i])
        #pl.imshow(X[i])
        plt.show()
        plt.imshow(np.reshape(Y,[batch_size,patch_h,patch_w,num_lesion_class+1])[i,:,:,0],cmap='gray')
        plt.show()

patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
patch_h=patch_height
patch_w=patch_width
N_subimgs = int(config.get('training settings', 'N_subimgs'))
inside_FOV = config.getboolean('training settings', 'inside_FOV')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
name_experiment = config.get('experiment name', 'name')
filename='./DRIVE/training/train'
best_last = config.get('testing settings', 'best_last')
path_experiment = './' +name_experiment +'/'
N_imgs=int(config.get('training settings', 'full_images_to_train'))

path_data = config.get('data paths', 'path_local')
DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
DRIVE_train_groundTruth = path_data + config.get('data paths', 'train_groundTruth')
train_imgs_original = load_hdf5(DRIVE_train_imgs_original)#[img_id:img_id+1]
train_masks=np.zeros([N_imgs,1,train_imgs_original.shape[2],train_imgs_original.shape[3]])
train_masks = load_hdf5(DRIVE_train_groundTruth )#masks always the same

mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]

Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
DRIVE_test_groundTruth = path_data + config.get('data paths', 'test_groundTruth')
test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
test_masks=np.zeros([Imgs_to_test,1,train_imgs_original.shape[2],train_imgs_original.shape[3]])
test_masks = load_hdf5(DRIVE_test_groundTruth )#masks always the same

train_imgs = my_PreProc(train_imgs_original)
#train_imgs = train_imgs_original
train_masks = train_masks/255.
train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565

test_imgs = my_PreProc(test_imgs_original)

test_masks = test_masks/255.
test_imgs = test_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
test_masks = test_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565

class_weight=np.array([20.0])#,60.0],20.0,90.0])#[10.0,30.0,20.0,60.0])#[10.0,30.0,20.0,60.0]
class_weight=class_weight/np.sum(class_weight)
test_class_weight=np.array([1.0])#,0.0,0.0,1.0])


mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]#,

gen=Active_Generate(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs,class_weight,mlist)

test_mlist=[np.where(test_masks[:,0,:,:]==np.max(test_masks[:,0,:,:]))]#,

test_gen=Active_Generate(test_imgs,test_masks,patch_height,patch_width,batch_size,N_subimgs,Imgs_to_test,test_class_weight,test_mlist)

if NewTrain:
    model = get_unet(3, patch_height, patch_width)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])#['categorical_accuracy'])

    json_string = model.to_json()
    open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
elif ChangeReg:
    model = get_unet(3, patch_height, patch_width)
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    json_string = model.to_json()
    open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
else:
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])#'categorical_accuracy'])


checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5',
                               verbose=1,
                               monitor='val_loss',
                               mode='auto',
                               save_best_only=True) #save at each epoch if the validation decreased

val_num=76800

history = LossHistory()
hist=model.fit_generator(gen,#generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs),
                    epochs=N_epochs,
                    steps_per_epoch=N_subimgs*N_imgs/batch_size,
                    verbose=1,
                    callbacks=[checkpointer,history],
                    validation_data=test_gen,
                    validation_steps=N_subimgs*Imgs_to_test/batch_size)

history.loss_plot('epoch')
with open('loss_plot.txt','w') as f:
    f.write(str(hist.history))

model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)

