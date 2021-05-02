import os
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.init as init
import random
import pandas as pd

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
#import pickle
import pickle
import warnings
import json
from pprint import pprint
import scipy.io as io
import PIL #added PIL
from PIL import Image
import pandas
from skimage.transform import resize
from torch.utils import data
#from other file
import numbers
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
# import Augmentor
from skimage.util.noise import random_noise
import scipy.misc

net = 1 #change to 0 for u-net with bach norm, 1: DnCNN

test_type = 1
CSV_path = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/SR_training/'

class DataGenerator(data.Dataset):
    def __init__(self, num_exps=100, batch_size=4, dim=(128,128), n_channels=1, shuffle=True, train = True, validation = False, test = False, test_type=1, transform=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.num_exps = num_exps
        self.train = train
        self.validation = validation
        self.test = test
        self.on_epoch_end()
        self.test_type = test_type
        self.transform = transform

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_exps / self.batch_size))

    def __getitem__(self, index): #index = batch num
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.train == False and self.validation == False and self.test == True:
            in1 = np.array([0])
            in2 = in1.tolist()
            indexes = in2
            #print('ind here: ',indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = indexes#[self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        #if self.transform:
        #    X, Y = ToTensor()

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_exps)
        if self.train == True:
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(((self.batch_size)*4, self.n_channels, *self.dim)) #total  batch size here includes slices
        Y = np.empty(((self.batch_size)*4, self.n_channels, *self.dim))
        if self.train == True and self.validation == False and self.test == False:
            df = pandas.read_csv(CSV_path + 'samples_train_750_images.csv')
        if self.train == False and self.validation == True and self.test == False:
            df = pandas.read_csv(CSV_path + 'samples_test.csv')
        if self.train == False and self.validation == False and self.test == True:
            if self.test_type == 1:
                df = pandas.read_csv(CSV_path + 'samples_test_1_image_inference.csv')
            if self.test_type == 2:
                df = pandas.read_csv(CSV_path + 'samples_test_inference2.csv')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            x_img = np.asarray(PIL.Image.open(df['Images'][ID]))
            y_label = np.asarray(PIL.Image.open(df['Labels'][ID]))
            #print('X values', x_img[100:120,100:120])#[0,100:120,100:120])
            #print('train file name >>>>>>', df['Images'][ID])
            
            x_img = resize(x_img, (256, 256), mode='constant', preserve_range=True) #actual  image and label
            y_label = resize(y_label, (256, 256), mode='constant', preserve_range=True)#actual  image and label
            
            #slices here of 128x128
            imx1 = np.array([x_img[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,x_img.shape[0],self.dim[0]) for y in range(0,x_img.shape[1],self.dim[1])])
            imx1 = np.array(imx1)
            #print('>>>> ',imx1.shape)
            lbx1 = np.array([y_label[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,y_label.shape[0],self.dim[0]) for y in range(0,y_label.shape[1],self.dim[1])])
            #print('>>>>>>>>>>>>>>> ',X.shape)
            
            X[i*4:(i+1)*4, 0, ...] = imx1 / 65535 - 0.5
            #print('XXXXXX ',X.shape)
            Y[i*4:(i+1)*4, 0, ...] = lbx1 / 65535 - 0.5    #.squeeze()

        return torch.from_numpy(X), torch.from_numpy(Y)
        
#adding this class to convert numpy images in to the Tensors        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        noisy_image, clean_image = sample['X'], sample['Y']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(noisy_image),
                'landmarks': torch.from_numpy(clean_image)}        

def load_dataset():
    batch_size = 4
    batch_size_test = 1
    im_height = 128 #M
    im_width = 128  #N
    num_exps_train = 750 #total train examples = 57000 we are using all FOV 9 to 16 here which is 8*50 = 400 images + FOV from 1-7 also (7*50 = 350)
    num_exps_validation = 40  #total test examples = 3000
    num_exps_test = 1 #test dataset with raw data #1 FOV 8 image
            
    params_train = {'dim': (im_height,im_width),'num_exps':num_exps_train,'batch_size': batch_size,'n_channels': 1,'shuffle': True, 'train': True, 'validation': False, 'test': False, 'test_type':test_type, 'transform': ToTensor}
    #params_validation= {'dim': (im_height,im_width),'num_exps':num_exps_validation,'batch_size': batch_size,'n_channels': 1,'shuffle': False,'train': False, 'validation': True, 'test': False, 'test_type':test_type, 'transform': ToTensor}
            #test_set =1 raw data, 2 -> avg2 , 3-> avg4, 4 -> avg8, 5-> avg16, 6-> all together noise levels (1,2,4,8,16)
    params_test = {'dim': (im_height,im_width),'num_exps':num_exps_test,'batch_size': batch_size_test,'n_channels': 1,'shuffle': False, 'train': False,  'validation': False, 'test': True, 'test_type':test_type, 'transform': ToTensor}

    # training_generator = DataGenerator( **params_train)
    # validation_generator = DataGenerator(**params_validation)
    # test_generator = DataGenerator(**params_test)

    transformed_dataset_train = DataGenerator(**params_train) #convert to tensor 
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=1, shuffle=True, num_workers=4) #conver to data loader

    #train_loader = training_generator
    #print('len of train loader',len(dataloader_train))
    #print('shape of train loader',dataloader_train)
    #test_loader = test_generator
    transformed_dataset_test = DataGenerator(**params_test) #convert to tensor 
    dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=4) #conver to data loader
    #print('len of test loader',len(dataloader_test))
    #from here onwards the model is added
    return dataloader_train, dataloader_test