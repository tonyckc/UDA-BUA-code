# Unsupervised Domain Adaptation for Low-dose CT Reconstruction via Bayesian Uncertainty Alignment


This is the official PyTorch implementation of our paper, coined [Unsupervised Domain Adaptation for Low-dose CT Reconstruction via Bayesian Uncertainty Alignment] ([https://ieeexplore.ieee.org/document/10593806](https://ieeexplore.ieee.org/document/10593806)). If you have any problems, please email me (ck.ee@my.cityu.edu.hk).


# Usage
## Environment  
please see the requirement file
## Dataset 
     - Source Domain: 2016 NIH AAPM-Mayo Clinic Low-dose CT Grand Challenge Dataset (25% of normal dose)
     - Target Domain: 
          - Simulated 5% dose of source domain
          - Simulated 50% dose of source domain
          - ISICDM 2020 Challenge Dataset (<25% dose)

Due to the restricted license of AAPM-Moyo dataset, we CAN NOT directly share the above datasets with you. Two options for you:

**Get the datasets from our:** - You can get the license for the AAPM-Moyo dataset by following the instructions at https://www.aapm.org/grandchallenge/lowdosect/ . If you have get the access right from AAPM, you can email us (ck.ee@my.cityu.edu.hk) to provide the given proof. Then, we will share with you a download link including all the above datasets. 

**Simulate by yourself:** - We provide a simulation file that can be used to process your dataset.

## Training your model
1. download the Fundus dataset at https://drive.google.com/file/d/1zTeTiTA5CBKOCPq_xVRajWVKUtjjPSrF/view?usp=sharing and put it into the dir at "your_path/fundus/*"
2. create the environment as required by the requirement file
3. set key training parameters for train.py file:
     - "datasetTrain" - source domains, such as [1,2,4]
     - "datasetTest" - target domain, such as [3]
     - "data-dir" - where your dataset as step 1
     - "label" - determine the objective ( OC or OD ?)of validation for best model choice.
4. run train.py and you will get the saved model
## Visualization using saved model
1. set the you wanted model dir in test_visulization.py 
2. run test_visualization.py

## Demo: Fast testing on a target domain using the provided .ckpt model
We offered a .ckpt file at https://drive.google.com/file/d/1-ntNwztBANmKnkf6VBZqEWGPqYL5sWg-/view?usp=sharing (OD - Target domain=4 - ASD=0.831 - Dice=0.936 ). You can use this model to get a fast testing process on the OD #4 target domain using test_visualization.py  
