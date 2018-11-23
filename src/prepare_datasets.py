import sys
sys.path.append('./lib')
import os,numpy as np,cv2,h5py,configparser,glob
from PIL import Image
from math import ceil
from help_functions import *

config = configparser.RawConfigParser()
config.read('./configuration.txt')

N_epochs = int(config.get('training settings', 'n_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
#train
original_imgs_train = config.get('data paths', 'original_imgs_train')
print(original_imgs_train)
groundTruth_imgs_train = config.get('data paths', 'groundTruth_imgs_train')
num_lesion_class = int(config.get('data attributes','num_lesion_class'))
#test
original_imgs_test = config.get('data paths', 'original_imgs_test')
groundTruth_imgs_test = config.get('data paths', 'groundTruth_imgs_test')

dataset_path = config.get('data paths','dataset_path')

print ("HINT: all images must have the same size!")
imgList = glob.glob(original_imgs_train + '/*.bmp')
Nimgs_train=len(imgList)
config.set('data attributes','total_data_train',str(Nimgs_train))

imgList = glob.glob(original_imgs_test + '/*.bmp')
Nimgs_test=len(imgList)
config.set('data attributes','total_data_test',str(Nimgs_test))


img2test=cv2.imread(imgList[0])

channels = img2test.shape[2]
height = img2test.shape[0]
width = img2test.shape[1]
config.set('data attributes','height',str(height))
config.set('data attributes','width',str(width))
config.write(open('configuration.txt','w'))


def get_datasets(imgs_dir,groundTruth_dir,Nimgs,train_test="null"):
    nameLength = int(len(str(Nimgs))+1)
    print (nameLength)
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,1,height,width))

    #border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir):
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            img = img.resize([width,height],Image.BICUBIC)
            imgs[i] = np.asarray(img)

            groundTruth_name = files[i][0:nameLength] + "manual"+str(num_lesion_class+1)+".gif"#gif"   #####+1
            print ("ground truth %d name: %s"%(num_lesion_class , groundTruth_name))
            g_truth = Image.open(groundTruth_dir + groundTruth_name).resize([width,height],Image.BICUBIC).convert('L')
            groundTruth[i,0,:,:] = np.asarray(g_truth)
            #corresponding border masks
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 )
    assert(np.min(groundTruth)==0 )
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))

    #assert(groundTruth.shape == (Nimgs,1,height,width))
    #assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth

imgs_train, groundTruth = get_datasets(original_imgs_train,groundTruth_imgs_train,Nimgs_train,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")

imgs_test, groundTruth = get_datasets(original_imgs_test,groundTruth_imgs_test,Nimgs_test,"test")
print ("saving test datasets")

write_hdf5(imgs_test, dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
