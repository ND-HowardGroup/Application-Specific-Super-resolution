# Super-resolution CNN based on DenseED blocks
This repository includes the code used for the geenration of super-resolution of microscopy images training using the small data (a small training dataset and really useful for biomedical applications like X-ray, MRI and in vivo imaging). 

Datasets: 
1. W2S dataset (open-source dataset)
2. BPAE dataset (custom-built two-photon microscopy)

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
