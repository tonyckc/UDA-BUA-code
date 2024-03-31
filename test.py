'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-22 10:30:28
@LastEditors: GuoYi
'''
import random
import torch
import os
import sys
from numpy import *
import numpy as np
import pydicom
import numpy as np

import time
from torch.optim import lr_scheduler
from torch import optim

from utils import check_dir
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets_DA import BuildDataSet,BuildDataSet_n2n,get_test,get_test_n2n,get_test_uda
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE,Baseline,DA_Denoiser,WGAN,RFM,RedCNN,cyclegan,noiser2noise,disc,UDA

from test_function import test_model,test_model_ours
from exhibit_function import model_updata
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
class InitParser(object):
    def __init__(self):
        self.gpu_id = 3
        self.version = "v3"
        self.mode = "test"
        self.batch_size= {"train": 20, "val": 20, "test": 1}
        self.model_index = 275
        self.source = 'AAPM'
        self.test = True
        self.validate = False
        # if use show datasets
        self.show = True
        # if save fig
        self.plot = True
        self.print_each_value = True
        self.target = 'PH'
        self.baseline = False
        self.baseline_type =  'baseline'#'uda'#'clycle'#'uda'#'n2n'#'uda'
        self.noise_intensity = 1
        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 20

        self.re_load = False

        self.is_shuffle = True if self.mode is "train" else False
        self.data_length = {"train":5000, "val":500, "test":4}
        batch_num = {x:int(self.data_length[x]/self.batch_size[x]) for x in ["train", "val", "test"]}
        self.show_batch_num = {x:int(batch_num[x]/10) for x in ["train", "val", "test"]}
        self.n_d_train = 1  ## Every training the generator, training the discriminator n times
        self.n_g_train = 2
        self.use_p_loss = True
        self.re_load = False
        ############################
        #### Important settings#####
        ############################
        self.gp = True
        self.xt_lambda = 0
        self.num_mc = 10
        self.bayesian_coral = 10  # bayesian_coral
        self.posterior_rho_init = -4
        self.mope = True
        self.mope_delta = 0.1
        self.style_lambda = 0.001
        self.seed = 0
        self.metric = False
        self.lsgan = False
        self.remote = True
        self.estimate_coral = True
        #####################
        self.coral = True
        self.style = False
        self.validate_coral = False
        #####################
        self.xt_reconstruction = True
        self.MAE = True
        self.unpaired = False
        self.bayesian_decoder = True
        self.bayesian_encoder = False
        if self.baseline:
            print('-' * 20, 'This is Baseline Model', '-' * 20)
            print('-' * 20)
        else:
            print('-' * 20, 'This is not Baseline Model', '-' * 20)
            print('-' * 20)
        print('-' * 20, 'Source Domain:{}---'.format(self.source), 'Target Domain:{}'.format(self.target),
              '-' * 20)
        ### set the parameter for loss
        self.p_lambda = 1
        self.m_gamma = 1
        # path setting
        if torch.cuda.is_available():
            self.data_root_path = "/hdd/ckc/CT/Dataset_npy" # "/media/ubuntu/88692f9c-d324-4895-8764-1cf202f9e6ac/chenkecheng/AAPM-challenge" #/home/dataset
            # save the result path
            self.root_path = "/hdd/ckc/CT/DA-CT"  # /home/dataset
        else:
            self.data_root_path = "V:/users/gy/data/Mayo/mayo_mat"
            self.root_path = "V:/users/gy/MyProject/WGAN_SACNN_AE"
        self.model_name = "WGAN_SACNN_AE_E"
        self.direct = True
        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/{}".format(self.version)
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        # self.train_folder = ["L192","L286","L291","L310","L333", "L506"]
        # self.test_folder = ["L067", "L096","L109","L143"]
        # self.val_folder = ["L067", "L096","L109","L143"]
        self.train_folder = ["L192"]
        self.test_folder = ["L067"]
        self.val_folder = ["L067"]
        self.source_weight = 0.01
        self.xt_reconstruction_weight = 0.01
        if self.re_load or self.mode is "test":
            self.old_version = "v2"
            self.old_result_path = self.root_path + "/results/{}".format(self.old_version)
            self.old_modle_path = self.old_result_path + "/model"
            self.old_modle_name = self.model_name + str(299) + "_val_Best"

        import os
        self.image_save = self.result_path + '/{}/{}2{}'.format(self.baseline_type,self.source,self.target)
        if not os.path.exists(self.image_save):
            os.makedirs(self.image_save)
        else:
            print('path exist')


def main(args):


    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        args.seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        args.seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #if args.baseline == True and args.baseline_type == 'n2n':
    #    datasets = {"test": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
    #                                     args.val_folder, False, None, args.data_length["test"], "show")}
    #else:
    #    datasets = {"test": BuildDataSet(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
    #                                args.val_folder, False, None, args.data_length["test"], "show")}
    if args.baseline == True and args.baseline_type == 'n2n':
        datasets = {"test": get_test_n2n(args.data_root_path,region=args.validate_region,show=args.show,noise=args.noise_intensity)}
    elif args.baseline == True and args.baseline_type == 'uda':
        datasets = {"test": get_test_uda(args.data_root_path, region=args.validate_region,show=args.show)}
    else:
        datasets = {"test": get_test(args.data_root_path,region=args.target,show=args.show)}
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = datasets#{x: DataLoader(datasets[x], args.batch_size[x], shuffle=True, **kwargs) for x in ["test"]}

    print("Load Datasets Done")

    ## *********************************************************************************************************
    #model_index = args.model_index
    #model = WGAN_SACNN_AE(args.batch_size[args.mode], args.root_path, args.version)
    #model = model_updata(model, model_old_name=args.model_name + "{}".format(model_index), model_old_path=args.model_path)
    print(args.baseline)
    if args.baseline:
        if args.baseline_type == 'WGAN':
            model = WGAN(args, device=device)
        elif args.baseline_type == 'RFM':
            print('RFM')
            model = RFM(args, device=device)
        elif args.baseline_type == 'RedCNN':
            model = RedCNN(args, device=device)
        elif args.baseline_type == 'clycle':
            print('clycle')
            model = cyclegan(args, device=device)
        elif args.baseline_type == 'n2n':
            print('n2n')
            model = noiser2noise(args, device=device)
        elif args.baseline_type == 'disc':
            model = disc(args, device=device)
        elif args.baseline_type == 'uda':
            print('uda')
            model = UDA(args, device=device)
        else:
            model = Baseline(args, device=device)
    else:

        model = DA_Denoiser(args,device=device)

    #

    print('-'*10, 'start to load model','-'*10)
    #pretrained_generator = torch.load(
    #    '/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/search/CoRAL-10-bayesianDecoder-2022-12-25 21:29:47.993116seed_0_mc_10_xtRe_0_Coral_10_mope_delta_0.1_posterior_rho_-4/model/ours-90.pkl')
    names =  [80]#,60,70,80,90,100[100,110,120,130,140,150,160,170,180,190,200]
    coral_value = 5
    if args.show == True:
        experiments_num = 1
    else:
        experiments_num = 3
    for name in names:
            psnr = []
            ssim = []
            vis = []
            vif = []
            for num in range(experiments_num):
                print('-'*10)
                if args.baseline == True and args.baseline_type == 'baseline':
                    pretrained_generator = torch.load(
                        '/hdd/ckc/CT/DA-CT/results_MAE_final/training_on_PH_validate_on_AAPM_new_region/Baseline-None-p=1-m=1-2023-06-27 11:03:41.365502seed_0_mc_9999_xtRe_0_Coral_10_mope_delta_0.1_posterior_rho_-4/model/Baseline-None-p=1-m=1-100.pkl')

                elif args.baseline == True and args.baseline_type == 'RFM':
                    pretrained_generator = torch.load(
                        '/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/search/Baseline-RFM-2022-12-26 13:25:02.683963seed_0_mc_9999_xtRe_0_Coral_5_mope_delta_0.1_posterior_rho_-4/model/Baseline-{}--{}.pkl'.format(args.baseline_type,
                            name))
                elif args.baseline == True and args.baseline_type == 'n2n':
                    pretrained_generator = torch.load(
                        '/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/search/Baseline-n2n-2022-12-26 13:46:35.708036seed_0_mc_9999_xtRe_0_Coral_5_mope_delta_0.1_posterior_rho_-4/model/Baseline-{}--{}.pkl'.format(args.baseline_type,
                            name))
                elif args.baseline == True and args.baseline_type == 'clycle':
                    pretrained_generator = torch.load(
                        '/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/search/Baseline-clycle-20-0.5-2023-01-01 23:23:06.832019seed_0_mc_9999_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/Baseline-{}-20-0.5--{}.pkl'.format(
                            args.baseline_type,
                            name))
                elif args.baseline == True and args.baseline_type == 'uda':
                    pretrained_generator = torch.load(
                        '/hdd/ckc/CT/DA-CT/results_MAE_final/training_on_AAPM_validate_on_AAPM_50_new_region/unpaired=False/CoRAL-1+Style-1-2023-07-03 21:42:32.064076xsRe_0.01_mc_10_xtRe_1_Coral_1_mope_delta_0.1_posterior_rho_-4/model/CoRAL-1+Style-1-9.pkl'.format(
                            args.baseline_type,
                            name))

                else:
                    print('load ours')
                    print('ckc')
                    #pretrained_generator = torch.load(
                    #    '/root/autodl-tmp/results_MAE_final/training_on_abd_validate_on_head_new_region/search/CoRAL-{}/model/CoRAL-{}-bayesianDecoder--{}.pkl'.format(coral_value,coral_value,name))
                    #pretrained_generator = torch.load(
                    #'/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/ablation/Style-0.001-bayesianDecoder-lsgan-D-2-2023-01-03 15:47:50.876592seed_0_mc_1_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/Style-0.001-bayesianDecoder-lsgan-D-2--{}.pkl'.format(name))
                    pretrained_generator = torch.load('/hdd/ckc/CT/DA-CT/results_MAE_final/training_on_AAPM_validate_on_AAPM_new_region/Baseline-None-p=1-m=1-2023-06-26 15:37:27.490234seed_0_mc_9999_xtRe_0_Coral_10_mope_delta_0.1_posterior_rho_-4/model/Baseline-None-p=1-m=1-100.pkl', map_location = lambda storage, loc: storage)
                    #pretrained_generator = torch.load(
                    # '/root/autodl-tmp/results_MAE_final/training_on_head_validate_on_abd_new_region/search/CoRAL-{}/model/CoRAL-{}-bayesianDecoder--{}.pkl'.format(coral_value,coral_value,name))

                if args.baseline == True and args.baseline_type == 'clycle':
                    model_dict_generator = model.netG_A.state_dict()
                    # pretrained_dict = {k.replace('generator.', ''): v for k, v in pretrained_generator.items() if
                    #                   k in model_dict_generator}
                    pretrained_dict = {k: v for k, v in pretrained_generator.items()}
                    model_dict_generator.update(pretrained_dict)
                    model.netG_A.load_state_dict(model_dict_generator)
                else:
                    model_dict_generator = model.generator.state_dict()
                    #pretrained_dict = {k.replace('generator.', ''): v for k, v in pretrained_generator.items() if
                    #                   k in model_dict_generator}
                    pretrained_dict = {k: v for k, v in pretrained_generator.items() if k in model_dict_generator}
                    model_dict_generator.update(pretrained_dict)
                    model.generator.load_state_dict(model_dict_generator)
                #print('-' * 10, '...loading generator sucecuss...', '-' * 10)
                model.to(device)
                ## *********************************************************************************************************
                mean_psnr,mean_ssim,mean_vif,mean_vis = test_model_ours(model = model,
                        dataloaders = dataloaders,
                        args = args)

                psnr.append(mean_psnr)
                ssim.append(mean_ssim)
                vif.append(mean_vif)
                vis.append(mean_vis)
            print(name)
            print('psnr',mean(psnr),std(psnr))
            print('ssim', mean(ssim), std(ssim))
            print('DSS', mean(vif), std(vif))
            print('GMSD', mean(vis), std(vis))

if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")




