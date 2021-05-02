import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
from model.dense_ed import DenseED
import random
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

#import pickle
import pickle
import warnings
import os
import json
from pprint import pprint
import scipy.io as io
import PIL #added PIL
from PIL import Image
import pandas
from skimage.transform import resize

from torch.utils import data

#from other file
import os
import numpy as np
from PIL import Image
import numbers
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
import sys
# import Augmentor
from skimage.util.noise import random_noise
import scipy.misc
plt.switch_backend('agg')
import torch
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cpu:0'))

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

net = 1 #change to 0 for u-net with bach norm, 1: DnCNN

test_type = 1
CSV_path = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Spring21/Feb21/0602/SR_training/vm521/'
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
            df = pandas.read_csv(CSV_path + 'samples_train_all.csv')
        if self.train == False and self.validation == True and self.test == False:
            df = pandas.read_csv(CSV_path + 'samples_test.csv')
        if self.train == False and self.validation == False and self.test == True:
            if self.test_type == 1:
                df = pandas.read_csv(CSV_path + 'samples_test_inference_short.csv')
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

batch_size = 4
batch_size_test = 1
im_height = 128 #M
im_width = 128  #N
num_exps_train = 400 #total train examples = 57000 we are using all FOV 9 to 16 here which is 8*50 = 400 images
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
import torch
import torch.nn as nn
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


        
#training imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import torch
import json
import random
import time
from pprint import pprint
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#load the model here
in_channels = 1
out_channels = 1
#device = '/CPU:0'
lr = 1e-4 #learning rate
wd = 1e-4 #weight decay
logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
n_train_pixels = batch_size*im_height*im_width*4 #slices
n_test_pixels = batch_size_test*im_height*im_width*4 #here batch is only 1 sample

#model selction based on net value
# if net == 0: #unet batch norm
#     model = UnetN2N(in_channels, out_channels).to(device) #set the model and assign to the device too
# if net == 1: #dncnn
#     depth = 17
#     width = 64
#     model = DnCNN(depth=depth, n_channels=width, image_channels=1, use_bnorm=True, kernel_size=3).to(device) #DnCNN model

model = DenseED(in_channels=1, out_channels=1, 
                blocks=[8,8,8],
                growth_rate=16, 
                init_features=64,
                drop_rate=0.38,
                bn_size=8,
                bottleneck=False,
                out_activation=None).to(device)

# optim_parameters = {'lr': lr}
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, betas=[0.9, 0.99], **optim_parameters)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3,
                       weight_decay=3e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)
print('DenseED Model summary >>>>>>>> ',model) #diasblaed model print
# print('DnCNN Model size >>>>>>> ',model.model_size)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

train_len = np.floor(len(dataloader_train)/batch_size)
#print('train step size', train_len)
test_len = np.ceil(len(dataloader_test)/batch_size)
#print('test step size', test_len)
ckpt_dir = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Spring21/Feb21/0602/SR_training/vm521/saved_model/'
#train function
def generateNumber(num):
    mylist = []
    for i in range(num):
        mylist.append(i)
    return mylist


def train(epochs):
    iters = 0
    mse_loss_train = []
    mse_loss_test = []
    total_steps = epochs * len(dataloader_train)
    for epoch in range(1, epochs + 1):
        model.train()
        mse=0
        #mse_sample = 0
        for batch_idx, (noisy_input, clean_target) in enumerate(dataloader_train):
            iters += 1
            #print('Input shape here >>>>>>', noisy_input.shape)
            #noisy_input = torch.from_numpy(noisy_input)
            #clean_target = torch.from_numpy(clean_target)
            noisy_input = torch.squeeze(noisy_input, 0) #from 5D to 4D conversion
            clean_target = torch.squeeze(clean_target, 0) #from 5D to 4D conversion
            noisy_input = noisy_input.float()
            clean_target = clean_target.float()
            noisy_input, clean_target = noisy_input.to(device), \
                clean_target.to(device)
            
            model.zero_grad()
            denoised = model(noisy_input)
            
            #convert to 256x256 image
            full_size = 256 
            part_size = 128
            denoised_full = torch.zeros(batch_size, 1, full_size, full_size)
            clean_full = torch.zeros(batch_size, 1, full_size, full_size)
            for i in range(batch_size):
                #print('xxxxx >>>>', i)
                denoised_full[i, 0, 0:part_size, 0:part_size] = denoised[i*4,:,:,:]
                denoised_full[i, 0, 0:part_size, part_size:full_size] = denoised[(i*4)+1,:,:,:]
                denoised_full[i, 0, part_size:full_size, 0:part_size] = denoised[(i*4)+2,:,:,:]
                denoised_full[i, 0, part_size:full_size,part_size:full_size] = denoised[(i*4)+3,:,:,:]
                
                clean_full[i, 0, 0:part_size, 0:part_size] = clean_target[(i*4),:,:,:]
                clean_full[i, 0, 0:part_size, part_size:full_size] = clean_target[(i*4)+1,:,:,:]
                clean_full[i, 0, part_size:full_size, 0:part_size] = clean_target[(i*4)+2,:,:,:]
                clean_full[i, 0, part_size:full_size,part_size:full_size] = clean_target[(i*4)+3,:,:,:]
                
            loss = F.mse_loss(denoised_full, clean_full, size_average=False) #this is the error bwtween images of 256x256 MSE loss
            loss.backward()

            step = epoch * len(dataloader_train) + batch_idx + 1
            pct = step / total_steps
            #lr = scheduler.step(pct)
            #adjust_learning_rate(optimizer, lr)
            optimizer.step()

            mse += loss.item()
            #mse_sample = loss.item() #this is the present loss
        rmse = np.sqrt(mse / n_train_pixels)
        scheduler.step(rmse)
        
        logger['rmse_train'].append(rmse)
        print("Epoch {} training RMSE: {:.6f}".format(epoch, rmse))
            #sys.exit(0)
        
        mse_loss_train.append(rmse)
        # test ------------------------------
        with torch.no_grad():
            model.eval()
            mse_test = 0.
            for batch_idx, (noisy, clean) in enumerate(dataloader_test):
                #noisy = torch.from_numpy(noisy)
                #clean = torch.from_numpy(clean)
                noisy = torch.squeeze(noisy, 0) #from 5D to 4D conversion
                clean = torch.squeeze(clean, 0) #from 5D to 4D conversion
                noisy = noisy.float()
                clean = clean.float()
                noisy, clean = noisy.to(device), clean.to(device)
                denoised = model(noisy)
                loss = F.mse_loss(denoised, clean, size_average=False)
                mse_test += loss.item()
                par_babu = denoised.cpu().detach().numpy() 
                tar_babu = clean.cpu().detach().numpy() 
                if epoch % 50 == 0:
                    io.savemat('target_pressure_%d.mat'%epoch, dict([('target_pressure1_test',np.array(tar_babu))]))
                    io.savemat('predict_pressure_%d.mat'%epoch, dict([('predict_pressure1_test',np.array(par_babu))])) 

            rmse_test = np.sqrt(mse_test/ n_test_pixels)
            logger['rmse_test'].append(rmse_test)
            print("Epoch {}: test RMSE: {:.6f}".format(epoch, rmse_test))
        mse_loss_test.append(rmse_test)
    
    #end of for loop so saving results here    
    timestr_plt = time.strftime("%Y%m%d_%H%M%S")
    np.savetxt(path+'/results/'+'DenseED_train_RMSE_0602_2021_'+timestr_plt+'.txt', np.array(mse_loss_train))
    np.savetxt(path+'/results/'+'DenseED_test_RMSE_0602_2021_'+timestr_plt+'.txt', np.array(mse_loss_test))
    # save model
    torch.save(model.state_dict(), ckpt_dir + "/DenseED_model_epoch{}.pth".format(epoch))
    torch.save(model,ckpt_dir+'DenseED_model_SR_0602_2021_'+timestr_plt+'.pt')
    #plots 
    xp = np.array(generateNumber(epochs))
    loss_train = np.array(mse_loss_train)
    loss_test = np.array(mse_loss_test)
    plt.plot(xp, loss_train, 'b*', label='Train_loss')
    plt.plot(xp, loss_test, 'rs', label='Test_loss')
    plt.xticks(np.arange(min(xp), max(xp)+1, 1.0))
    plt.xlabel('Epochs ')
    plt.ylabel('Train and Test Loss ')
    plt.title('Training SR dataset ')
    plt.legend(loc='best', frameon=False)
    
    plt.savefig(path+'results/'+'DenseED_RMSE_loss_0602_2021_'+timestr_plt+'.png')

if __name__ == "__main__":
    #Trainer(default_save_path=’/your/path/to/save/checkpoints’)
    path = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Spring21/Feb21/0602/SR_training/vm521/'
    #with open(path + "/args.txt", 'w') as args_file:
        #json.dump(logger, args_file, indent=4)
    Epochs =  200  
        #Trainer(default_save_path=path)
    print('Start training........................................................')
    t1 = time.time()
    train(Epochs)
    t2 = time.time()
    #torch.save(model_H,'ConvNet2311_Hadamard_03.pt')
    print('Execution time >>>>>>>:', t2-t1)