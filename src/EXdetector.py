#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import configparser
from matplotlib import pyplot as plt
import cv2
import glob
from keras.models import model_from_json
from keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')

# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_predict_overlap
# pre_processing.py
from pre_processing import my_PreProc
import xlrd
#from xlutils.copy import copy
#import xlutils
from numba import jit
@jit
def pixBYpix_fast(thresh,image):
    w=image.shape[0]
    h=image.shape[1]
    d=image.shape[2]
    for i in range(w):
        for j in range(h):
            for z in range(d):
                if image[i,j,z]>=thresh:
                    image[i,j,z]=1
                else:
                    image[i,j,z]=0
    return image



class medicalImages():
    def loadImage(self,imgs):
        self.image=imgs
        self.width=imgs.shape[1]
        self.height=imgs.shape[0]
        self.depth=imgs.shape[2]

    def normlize(self,imgs):
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs - imgs_mean) / imgs_std

        imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (
        np.max(imgs_normalized) - np.min(imgs_normalized))) * 255
        return imgs_normalized

    def clahe_equalize(self,imgs):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs.shape)
        imgs_equalized = clahe.apply(np.array(imgs, dtype=np.uint8))
        return imgs_equalized

    def adjust_gamma(self,imgs, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = np.empty(imgs.shape)
        new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
        return new_imgs

    def adjustColor(self):
        processed=np.zeros((self.height,self.width,self.depth))
        for i in range(self.depth):
            imgSplit=self.image[:,:,i]
            imgSplit= self.normlize(imgSplit)
            imgSplit = self.clahe_equalize(imgSplit)
            imgSplit = self.adjust_gamma(imgSplit)
            #self.imshow(imgSplit)
            processed[:,:,i]=imgSplit
        processed=processed.astype('uint8')
        return processed

config = configparser.RawConfigParser()
config.read('configuration.txt')
path_data = config.get('data paths', 'path_local')
#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')


#the border masks provided by the DRIVE
#DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
#test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
N_visual = int(config.get('testing settings', 'N_group_visual'))
average_mode = config.getboolean('testing settings', 'average_mode')
num_lesion_class = int(config.get('data attributes','num_lesion_class'))
Nimgs = int(config.get('data attributes','total_data_test'))

#testinput='./TestInput/'
#E:\unet\retinaNN(EX)\DRIVE\test\images
#E:\unet\retinaNN(EX)\TestInput
testinput='./DRIVE/test/images/'
#imgList = glob.glob(unicode('.\\TestInput\\*.bmp','utf-8').encode('gbk'))
imgList = glob.glob(testinput+'/*.bmp')

Nimgs_train=len(imgList)
print (Nimgs_train)
count=1
solver=medicalImages()
for i in imgList:
    test_imgs_original = plt.imread(i)
    height, width = test_imgs_original.shape[:2]
    full_img_height = height#int(height*0.5)
    full_img_width = width#int(width*0.5)
    #size = (int(width*0.5), int(height*0.5))
    #test_imgs_original=cv2.resize(test_imgs_original, size,interpolation=cv2.INTER_AREA)
    name=i.split('\\')[-1]
    picname=name.split('.')[0]
    print (picname)
    patches_imgs_test = None
    new_height = None
    new_width = None

    patches_imgs_test, new_height, new_width = get_data_predict_overlap(
        imgPredict = test_imgs_original,  #original
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width,
        num_lesion = num_lesion_class,
        total_data = Nimgs
        )



#Run the prediction of the patches
    best_last = config.get('testing settings', 'best_last')
#Load the saved model
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=1)
    print ("\npredicted images size :")
    print (predictions.shape)

#Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions,"original")

    pred_patches=pred_patches.transpose(0,3,1,2)
    print (pred_patches.shape)


#Elaborate and visualize the predicted images
    pred_imgs = None
    orig_imgs = None
   
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions

    result=0
    #orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    print ("np max 0",np.max(pred_imgs[:,0,:,:])==1.0)
    #评级部分

    all=np.zeros((height*2, width,3))
    all[:height,:,:]=test_imgs_original

    pred_gt=pred_imgs.transpose(0,2,3,1)[0]
    pred_gt=pixBYpix_fast(0.5,pred_gt)
    all[height:2*height,:,:1]=(pred_gt[:,:,:1]*255)#.astype('uint8')
    all[height:2 * height, :, 1:2] = (pred_gt[:, :, :1] * 255)
    all[height:2 * height, :, 2:3] = (pred_gt[:, :, :1] * 255)
    img11 = Image.fromarray(np.uint8(all))
    img11.save('.\\TestResult\\'+name+'.jpg')
    print (i,result)
    
