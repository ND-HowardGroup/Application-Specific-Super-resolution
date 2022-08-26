# Super-resolution CNN based on DenseED blocks
This repository includes the code used for the geenration of super-resolution of microscopy images training using the small data (a small training dataset and really useful for biomedical applications like X-ray, MRI and in vivo imaging). 

Datasets: 
1. W2S dataset (open-source dataset)
2. BPAE dataset (custom-built two-photon microscopy)

Diffraction-limited image: (BPAE sample from test dataset: FOV8)

![](Results/Main_figure/main_images/DL_image.png)


Target super-resolution image geenrated with SRRF method: 

![](Results/Main_figure/main_images/target_SR_image.png)


Fully convolutional networks (FCNs) without DenseED blocks:

![](Results/Main_figure/main_images/FCN_no_denseED_Est_SR_image1.png)


Fully convolutional networks (FCNs) with DenseED blocks:

![](Results/Main_figure/main_images/FCN_denseED_Est_SR_image.png)


Generative adviseral networks (GANs) without DenseED blocks:

![](Results/Main_figure/main_images/simple_GANs_Est_SR_image_config9433.png)


Generative adviseral networks (GANs) with DenseED blocks:

![](Results/Main_figure/main_images/GANs_denseED_Est_SR_image.png)


Transfer learning: 
Mouse Kideny input: diffraction-limited image (FOV1)

![](Results/Transfer_learning_Mouse_kidney/DL_image_MK_orange.png)


Estiamted super-resolution image using FCN with DenseED blocks:

![](Results/Transfer_learning_Mouse_kidney/Est_SR_image_MK_orange.png)


Target super-resolution image (generated using SRRF method): 

![](Results/Transfer_learning_Mouse_kidney/target_SR_image_MK_orange.png)



## **Copyright**

© 2022 Varun Mannam, University of Notre Dame  

## **License**

Licensed under the [GPL](https://github.com/ND-HowardGroup/Deep_learning_Super-resolution/blob/main/LICENSE)
