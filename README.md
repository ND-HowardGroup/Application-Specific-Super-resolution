# Super-resolution CNN based on DenseED blocks
This repository includes the code used for the geenration of super-resolution of microscopy images training using the small data (a small training dataset and really useful for biomedical applications like X-ray, MRI and in vivo imaging). 

# Motivation: 
Figure shows the tradeoff between dataset size and performance: 
1. Show the requirement of the SR datasets
![](Results/motivation_files/big_motivation1.jpg)
2. Show the illustration of the ML model (tradeoff between training dataset size and performance metrics)
![](Results/motivation_files/Motivation1.jpg)
3.Tradeoff of ML methods performance metrics vs training dataset size
![](Results/motivation_files/motivation2.jpg)


# Datasets: 
1. W2S dataset (open-source dataset)
2. BPAE dataset (custom-built two-photon microscopy)

# Sample dataset images: 
1. W2S dataset samples (experimentally captured diffraction-limited (using widefield microscopy setup) and super-resolution images (SIM microscope setup))
![](Results/dataset_sample_images/training_dataset.jpg)
2. BPAE dataset samples (experimentally captured diffraction-limited (using two-photon microscopy setup) and super-resolution images (computationally generated using SRRF method))
![](Results/dataset_sample_images/training_dataset2.jpg)

# W2S dataset
Widefield2SIM dataset (comination of experimental diffraction-limited images and SIM images as target images of human cells)

# BPAE dataset
(BPAE sample from test dataset: FOV8)
Sample: BPAE, captured using custom-built two-photon microscopy

Diffraction-limited Image          | SRDenseED result         | Target super-resolution Image (using SRRF method)		         |	
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Results/Main_figure/main_images/DL_image.png" width="200" height="200" />   |  <img src="Results/Main_figure/main_images/FCN_denseED_Est_SR_image.png" width="200" height="200" />| <img src="Results/Main_figure/main_images/target_SR_image.png" width="200" height="200" /> |



# Gneralization of trained model (trained on BPAE dataset): 
Sample: Mouse Kidney, captured using custom-built two-photon microscopy

Diffraction-limited Image          | SRDenseED result         | Target super-resolution Image (using SRRF method)		         |	
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Results/Transfer_learning_Mouse_kidney/DL_image_MK_orange.png" width="200" height="200" />   |  <img src="Results/Transfer_learning_Mouse_kidney/Est_SR_image_MK_orange.png" width="200" height="200" />| <img src="Results/Transfer_learning_Mouse_kidney/target_SR_image_MK_orange.png" width="200" height="200" /> |




## **Copyright**

© 2022 Varun Mannam, University of Notre Dame  

## **License**

Licensed under the [GPL](https://github.com/ND-HowardGroup/Deep_learning_Super-resolution/blob/main/LICENSE)
