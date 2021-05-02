import os
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
from model.generative_network import Generator
from torch.autograd.variable import Variable

from model.discriminative_network import discriminative_network

import random
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
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
from load_dataset import load_dataset 
import numbers
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from skimage.util.noise import random_noise
import scipy.misc
plt.switch_backend('agg')
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cpu:0'))
from torch import nn, optim
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
#load dataset
from torchvision import models
from torchsummary import summary

dataloader_train, dataloader_test = load_dataset()


batch_size = 4
batch_size_test = 1
im_height = 128 #M
im_width = 128  #N
num_exps_train = 750 #total train examples = 57000 we are using all FOV 9 to 16 here which is 8*50 = 400 images + FOV from 1-7 also (7*50 = 350)
num_exps_validation = 40  #total test examples = 3000
num_exps_test = 1 #test dataset with raw data #1 FOV 8 image


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

#model is here
#load the generator
model_g = Generator().to(device)
model_d = discriminative_network().to(device)
#load the dis
print('generative network:',summary(model_g, (1, 128, 128)))
print('descriminator network:',summary(model_d, (1, 128, 128)))

# optim_parameters = {'lr': lr}
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, betas=[0.9, 0.99], **optim_parameters)
# # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-3,
#                        weight_decay=3e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
#                     verbose=True, threshold=0.0001, threshold_mode='rel',
#                     cooldown=0, min_lr=0, eps=1e-8)
optimizer_D = torch.optim.Adam(model_d.parameters(), lr=5e-4, weight_decay=5e-5)
optimizer_G = torch.optim.Adam(model_g.parameters(), lr=1e-5, weight_decay=1e-6)

# print('SR GANs Model summary >>>>>>>> ',model) #diasblaed model print
# print('DnCNN Model size >>>>>>> ',model.model_size)

gan_loss = nn.BCELoss()

train_len = np.floor(len(dataloader_train)/batch_size)
#print('train step size', train_len)
test_len = np.ceil(len(dataloader_test)/batch_size)
#print('test step size', test_len)
#ckpt_dir = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/SR_training/Model1_GANs/test_file/saved_model/'
#train function
def generateNumber(num):
    mylist = []
    for i in range(num):
        mylist.append(i)
    return mylist
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
def train(epochs):
    iters = 0
    mse_loss_train = []
    mse_loss_test = []
    g_losses = []
    d_losses = []
    model_d.train()
    model_g.train()   
    total_steps = epochs * len(dataloader_train)
    for epoch in range(1, epochs + 1):
        mse=0
        count =0
        #mse_sample = 0
        for batch_idx, (noisy_input, clean_target) in enumerate(dataloader_train):
            iters += 1
            count += 1 #no of batches here
            #print('Input shape here >>>>>>', noisy_input.shape)
            #noisy_input = torch.from_numpy(noisy_input)
            #clean_target = torch.from_numpy(clean_target)
            real = Variable(Tensor(16, 1).fill_(1), requires_grad=False)
            fake = Variable(Tensor(16, 1).fill_(0), requires_grad=False)
            noisy_input = torch.squeeze(noisy_input, 0) #from 5D to 4D conversion
            clean_target = torch.squeeze(clean_target, 0) #from 5D to 4D conversion
            noisy_input = noisy_input.float()
            clean_target = clean_target.float()
            noisy_input, clean_target = noisy_input.to(device), \
                clean_target.to(device)
            # print(clean_target.shape)
            # print(noisy_input.shape)
            # sys.exit(0)
            imgs_fake = Variable(model_g(noisy_input), requires_grad=False)
            optimizer_D.zero_grad()
            # denoised = model(noisy_input)
            # print(clean_target.shape)
            # print(imgs_fake.shape)
            d_loss = gan_loss(model_d(clean_target), real) + \
                    gan_loss(model_d(imgs_fake), fake)
            d_loss.backward()
            optimizer_D.step()



            imgs_fake = model_g(noisy_input)

            optimizer_G.zero_grad()

            # Minimizing (1-log(d(g(noise))) is less stable than maximizing log(d(g)) [*].
            # Since BCE loss is defined as a negative sum, maximizing [*] is == minimizing [*]'s negative.
            # Intuitively, how well does the G fool the D?
            g_loss = gan_loss(model_d(imgs_fake), real)
            mse_gloss = F.mse_loss(imgs_fake, clean_target, reduction='sum')

            new_loss = 0.6*mse_gloss+0.4*g_loss

            new_loss.backward()
            optimizer_G.step()
            denoised = imgs_fake


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
                
            loss = F.mse_loss(denoised_full, clean_full, reduction='sum') #this is the error bwtween images of 256x256 MSE loss replaced with reduction=sum from size_average= False
            # loss.backward()

            step = epoch * len(dataloader_train) + batch_idx + 1
            pct = step / total_steps
            #lr = scheduler.step(pct)
            #adjust_learning_rate(optimizer, lr)
            # optimizer.step()

            mse += loss.item()
            #mse_sample = loss.item() #this is the present loss
        rmse = np.sqrt(mse / (count*n_train_pixels)) #compensation for number of batches here
        # scheduler.step(rmse)
        
        logger['rmse_train'].append(rmse)
        print("Epoch {} training RMSE: {:.6f}".format(epoch, rmse))
            #sys.exit(0)
        
        mse_loss_train.append(rmse)
        # test ------------------------------
        with torch.no_grad():
            model_g.eval()
            mse_test = 0.
            count = 0
            for batch_idx, (noisy, clean) in enumerate(dataloader_test):
                #noisy = torch.from_numpy(noisy)
                #clean = torch.from_numpy(clean)
                count += 1 #no of batches here
                noisy = torch.squeeze(noisy, 0) #from 5D to 4D conversion
                clean = torch.squeeze(clean, 0) #from 5D to 4D conversion
                noisy = noisy.float()
                clean = clean.float()
                noisy, clean = noisy.to(device), clean.to(device)
                denoised = model_g(noisy)
                loss = F.mse_loss(denoised, clean, reduction='sum') #replaced with reduction=sum from size_average= False
                mse_test += loss.item()
                est_sr = denoised.cpu().detach().numpy() 
                tar_sr = clean.cpu().detach().numpy() 
                if epoch % 50 == 0:
                    io.savemat('target_super_resolution_%d.mat'%epoch, dict([('target_super_resolution_test',np.array(tar_sr))]))
                    io.savemat('predicted_super_resolution_%d.mat'%epoch, dict([('predicted_super_resolution_test',np.array(est_sr))])) 

            rmse_test = np.sqrt(mse_test/ (count*n_test_pixels)) #compensation for number of batches here
            logger['rmse_test'].append(rmse_test)
            print("Epoch {}: test RMSE: {:.6f}".format(epoch, rmse_test))
        mse_loss_test.append(rmse_test)
    
    #end of for loop so saving results here    
    timestr_plt = time.strftime("%Y%m%d_%H%M%S")
    np.savetxt(path+'results_GANs_train_RMSE_'+timestr_plt+'.txt', np.array(mse_loss_train))
    np.savetxt(path+'results_GANs_test_RMSE_'+timestr_plt+'.txt', np.array(mse_loss_test))
    # save model
    torch.save(model_g.state_dict(), path + "GANs_model_epoch{}.pth".format(epoch))
    torch.save(model_g,path+'GANs_model_SR_'+timestr_plt+'.pt')
    #plots 
    xp = np.array(generateNumber(epochs))
    loss_train = np.array(mse_loss_train)
    loss_test = np.array(mse_loss_test)
    plt.plot(xp, loss_train, 'b*', label='Train_loss')
    plt.plot(xp, loss_test, 'rs', label='Test_loss')
    plt.xticks(np.arange(min(xp), max(xp)+1, 1.0))
    plt.xlabel('Epochs ')
    plt.ylabel('Train and Test Loss ')
    plt.title('RMSE loss in GANs model (SR dataset) ')
    plt.legend(loc='best', frameon=False)
    
    plt.savefig(path+'results_GANs_RMSE_loss_'+timestr_plt+'.png')

if __name__ == "__main__":
    #Trainer(default_save_path=’/your/path/to/save/checkpoints’)
    path = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/SR_training/Model1_GANs/config9433/' #change the folder name every time
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