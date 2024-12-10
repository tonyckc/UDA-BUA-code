from torch import nn
import torch
import glob
import shutil
import os
import time
import numpy as np
import time
from glob import glob as file_glob
from torchvision import models
from datasets_DA import get_test_patch
import torch
from datetime import datetime


class InitParser(object):
    def __init__(self):
        self.gpu_id = 1
        self.seed = 0
        self.use_cuda = torch.cuda.is_available()
        self.mode = "train"
        self.patch_n = 1
        self.patch_size = 256
        self.source = 'AAPM'
        self.test = False
        self.validate = True

        self.target = 'AAPM_5'
        # unpaired means a setting, i.e., we can obtain the NDCT images from target domain, but the NDCT is unparied with LDCT.
        self.unpaired = False
        self.batch_size = {"train": 8, "val": 3, "test": 1}

        # source domain data length
        if self.source == 'PH':
            self.data_length = {"train": 843, "val": 0, "test": 0}  #
        elif self.source == 'AAPM':
            self.data_length = {"train": 5410, "val": 0, "test": 0}  #
        elif self.source == 'ISICDM':
            self.data_length = {"train": 3275, "val": 0, "test": 0}  #

        # target domain data length
        if self.target == 'PH':
            self.data_length["test"] = 281  # 1642
        elif self.target == 'ISICDM':
            self.data_length["test"] = 3275  # 1642
        elif self.target == 'AAPM':
            self.data_length["test"]= 526 # 1642

        # int(self.batch_size['train']/self.patch_n): determine loading how many images
        self.batch_size_dataloader = {"train": int(self.batch_size['train']/self.patch_n), "val": 3, "test": 1}
        # train the generator n_g_train times, and train the discriminator n_d_train times
        self.n_d_train = 2
        self.n_g_train = 1



        ##########################################################################
        #### model selectiom
        ##########################################################################
        # if is baseline model
        self.baseline = False
        # ours_new is our model, others are baseline for comparison
        self.baseline_type = 'ours_new' # ['clycle','uda','n2n','clycle','RFM','RFM','WGAN']
        # baseline models do not refer to the UDA, so they do not need to load the model
        self.re_load = False if self.baseline else True


        ###################################################
        ######## Source domain related settings
        ###################################################
        self.MAE = True
        # gradient penality for GAN
        self.gp = True
        # GAN type
        self.lsgan = True
        # beta1 in Eq (14), AAPM-A,AAPM-B,ISICDM-20: 1,0.01,0.1
        self.source_weight = 1
        # perception loss
        self.use_p_loss = True
        # perception loss weight
        self.p_lambda = 1
        self.m_gamma = 1

        ##################################################
        ####### Bayesian related parameters, which are fixed usually
        ##################################################
        self.posterior_rho_init = -4
        self.mope = True
        self.mope_delta = 0.1
        ####################################################
        ######  Bayesian Uncertainty Alignment
        ####################################################
        # coral means Bayesian CORAL, i.e. Bayesian Uncertainty Alignment
        self.coral = True
        # use estimated coral, rather than exact solution
        self.estimate_coral = True
        # check coral works or not, usually is False
        self.validate_coral = False
        # use the self-reconstruction loss in Eq. (9)
        self.xt_reconstruction = True
        #  # beta2 in Eq (14), AAPM-A,AAPM-B,ISICDM-20: 10,1,1
        self.bayesian_coral = 10  #
        # as in Eq (10) L_UR : L_SR = 1:1
        self.xt_reconstruction_weight = self.bayesian_coral



        #########################################################
        ####  Sharpness-aware Distribution Alignment via Adversarial Training
        #########################################################
        # L_SDA style is True, the adversarial loss is on
        self.style = True
        # when adversarial loss is on, using the MLV, direct should be False
        self.direct = False
        # beta3 in Eq (14), AAPM-A,AAPM-B,ISICDM-20: 100,1,0.1
        self.style_lambda = 100
        #####################



        print('-'*20,'This is Baseline Model','-'*20) if self.baseline else print('-' * 20, 'This is not Baseline Model', '-' * 20)
        print('-'*20,'Source Domain:{}---'.format(self.source),'Target Domain:{}'.format(self.target),'-'*20)

        self.num_workers = 0
        ## set optimizer
        self.lr = 1e-4
        if self.re_load:
            self.lr = self.lr * 0.5
        self.momentum = 0.9
        self.weight_decay = 0.0

        ## set training parameter
        self.epoch_num = 51
        self.validate_epoch = 1
        self.validate_shreshold = 101
        self.is_shuffle = True if self.mode is "train" else False
        times  = datetime.now()
        self.remote = True
        batch_num = {x:int(self.data_length[x]/self.batch_size[x]) for x in ["train", "val", "test"]}
        self.show_batch_num = {x:int(batch_num[x]/10) for x in ["train", "val", "test"]}

        # path setting
        # where dataset save
        self.data_root_path = "./*/Dataset_npy"
        # where training model save
        self.root_path = "./*/DA-CT"
        self.optimizer_g_name = "OptimG_E"
        self.optimizer_d_name = "OptimD_E"

        ## running name and how many MC sampling
        if self.baseline:
            # no MC
            self.num_mc = 9999
            self.version_name = 'Baseline-{}-p={}-m={}-'.format(self.baseline_type,self.p_lambda,self.m_gamma)
            self.version = self.version_name +  str(times)
        elif self.style == True and self.coral == False:
            self.version_name = 'Style-{}-bayesianDecoder-lsgan-D-2-'.format(self.style_lambda)
            self.version = self.version_name + str(times)
            # no MC
            self.num_mc = 1
        elif self.style == False and self.coral == True:
            #
            self.num_mc = 10
            self.version_name = 'CoRAL-{}-bayesianDecoder-'.format(self.bayesian_coral)
            self.version= self.version_name+ str(times)  #
        elif self.coral == True and self.style == True:
            self.num_mc = 10
            self.version_name = 'CoRAL-{}+Style-{}-'.format(self.bayesian_coral,self.style_lambda)
            self.version = self.version_name + str(times)  #
        else:
            self.num_mc = 1
            self.version_name = 'Only_XtR-'
            self.version = self.version_name + str(times)  #

        self.name = self.version + 'xsRe_{}_mc_{}_xtRe_{}_Coral_{}_mope_delta_{}_posterior_rho_{}'.format(self.source_weight,self.num_mc,self.xt_reconstruction_weight,self.bayesian_coral,self.mope_delta,self.posterior_rho_init)

        self.model_name = self.version_name
        # Test_baseline
        self.result_path = self.root_path + "/training_on_{}_validate_on_{}_new_region/unpaired={}/{}/{}/{}".format(self.source,self.target,self.unpaired,datetime.today().date(),self.baseline_type,self.name)
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.measure_path = self.result_path + "/measure"
        self.show_image = self.result_path + "/images"
        self.code_path = self.result_path + "/code"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        self.train_folder = ["L067","L096","L109","L143","L192", "L286","L291","L310"]
        self.test_folder = ["L333", "L506"]
        self.val_folder = ["L291"]

        if self.re_load or self.mode is "test":
            self.old_version = "v1"
            self.old_index = 315
            self.old_result_path = self.root_path + "/results/{}".format(self.old_version)
            self.old_modle_path = "./pretrained_model/Baseline-None-p=1-m=1-100.pkl"

            self.old_optimizer_path = self.old_result_path + "/optimizer"
            self.old_modle_name = self.model_name + str(self.old_index)
            self.old_optimizer_g_name = self.optimizer_g_name + str(self.old_index)
            self.old_optimizer_d_name = self.optimizer_d_name + str(self.old_index)

        
