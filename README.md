# Unsupervised Domain Adaptation for Low-dose CT Reconstruction via Bayesian Uncertainty Alignment


This is the official PyTorch implementation of our paper, coined [Unsupervised Domain Adaptation for Low-dose CT Reconstruction via Bayesian Uncertainty Alignment](https://ieeexplore.ieee.org/document/10593806)). 

If you have any problems, please email me (ck.ee@my.cityu.edu.hk).


# Usage
## Environment  
please see the requirement file
## Dataset 
     - Source Domain: 2016 NIH AAPM-Mayo Clinic Low-dose CT Grand Challenge Dataset (25% of normal dose)
     - Target Domain: 
          - Simulated 5% dose of source domain
          - Simulated 50% dose of source domain
          - ISICDM 2020 Challenge Dataset (<25% dose)

Due to the restricted license of AAPM-Moyo dataset, we **CANNOT** directly share the above datasets with you. Two options for you:

**Get the datasets from our:** - You can get the license for the AAPM-Moyo dataset by following the instructions at [AAPM Moyo challenge website](https://www.aapm.org/grandchallenge/lowdosect/) . If you have get the access right from AAPM, you can email us (ck.ee@my.cityu.edu.hk) to provide the given proof. Then, we will share with you a download link including all the above datasets. 

If you get the dataset,the dataset saving structure will be 

     - /your_path/
     
        └── AAPM
               ├── 1200000_1mm(50%)/
     
               ├── 110000_1mm (5%)/
     
               ├── full_1mm/
               
               ├── quarter_1mm (25%)/
          
        └── ISICDM
**Simulate by yourself:** We provide a simulation file that you can use to process your dataset.

## Pretrained Source Domain Model and VGG19 Model
1. We provide a source-trained model in 'pretrained_model/' folder. This model is trained on AAPM-Moyo 2016 dataset. You can also use your own pre-trained model. Please set the correct model path at [init.py](https://github.com/tonyckc/UDA-BUA-code/blob/main/init.py#L202).

2. We also provide the pretrained VGG19 model for perception loss computation at [here](https://drive.google.com/file/d/1mNAn0P42CcRx3-qzw76Px93s2H8jajc-/view?usp=sharing). Please download the pretrained model and put it to the 'pretrained_model/' folder

## Training
1. All hyperparameters are in [init.py](https://github.com/tonyckc/UDA-BUA-code/blob/main/init.py). Some important settings include
- "self.target": target domain, including  'AAPM_5', 'AAPM-50', and 'ISICDM'
- "self.baseline_type:" choose different models for training. Our model is named "ours_new" (corresponds to [DA_Denoiser in model.py](https://github.com/tonyckc/UDA-BUA-code/blob/main/model.py#L1478)). You can choose other baseline methods but need to set the [self.baseline = True](https://github.com/tonyckc/UDA-BUA-code/blob/main/init.py#L68)
- "self.data_root_path": path for dataset 
- "self.root_path": path for training results 
2. Run train.py

## LDCT image denoising/reconstruction benchmarking methods for unsupervised domain problems
We provide wide LDCT image denoising/reconstruction benchmarking methods for unsupervised domain problems. You can choose different model names at [self.baseline_type](https://github.com/tonyckc/UDA-BUA-code/blob/main/init.py#L68):
- [ClycleGAN](https://www.sciencedirect.com/science/article/pii/S1361841521002541): self.baseline_type = 'clycle'
- [Noise2noise](https://arxiv.org/abs/1803.04189): self.baseline_type = 'n2n'
- [CCDnet](https://www.sciencedirect.com/science/article/pii/S0010482523006844): self.baseline_type = 'RMF'
- [UDA](https://ieeexplore.ieee.org/abstract/document/9969607): self.baseline_type = 'uda'
- WGAN-VGG: self.baseline_type = 'WGAN'

***
If you find our code useful for your work please cite:
```
@article{kecheng2024tnnls,
  title={Unsupervised Domain Adaptation for Low-dose CT Reconstruction via Bayesian Uncertainty Alignment},
  author={Chen, Kecheng; Liu, Jie; Wan, Renjie; Lee, Victor; Vardhanabhuti, Varut; Yan, Hong; Li, Haoliang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
  
