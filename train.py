import torch
import random
import os
import sys
import numpy as np
import time
from torch.optim import lr_scheduler
from torch import optim

from init import InitParser
from utils import updata_model,save_file,chaeck_dir_all,set_seed
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets_DA import BuildDataSet,get_test,get_test_n2n,BuildDataSet_n2n,BuildDataSet_disc,BuildDataSet_UDA,BuildDataSet_unpaired
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE,Baseline,DA_Denoiser,WGAN,RFM,RedCNN,cyclegan,noiser2noise,disc,UDA,Baseline_deter
from train_function import train_model
import glob
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "CUDA is not available"
    # check if these dirs exist
    chaeck_dir_all([args.loss_path,args.model_path,args.optimizer_path,args.measure_path,args.show_image,args.code_path])
    # save code
    save_file(args.code_path)
    # set seed for good reproducation
    set_seed(args)

    # data augmentation
    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    # set different dataloader
    if args.baseline == True and args.baseline_type == 'n2n':
        datasets = {
            "train": BuildDataSet_n2n(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                  args.train_folder, True, pre_trans_img, args.data_length["train"], "train")}

    elif args.baseline == True and args.baseline_type == 'disc':
        datasets = {
            "train": BuildDataSet_disc(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train")}
    elif args.baseline == True and args.baseline_type == 'uda':
        datasets = {
            "train": BuildDataSet_UDA(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                  args.train_folder, True, pre_trans_img, args.data_length["train"], "train")}
    # our dataloader
    else:
        # upparied model can be ignored.
        if args.unpaired == True:
            datasets = {
                "train": BuildDataSet_unpaired(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train")}
        else:
            datasets = {"train": BuildDataSet(args.baseline,args.source,args.target, args.patch_n,args.data_root_path, args.train_folder, True, pre_trans_img, args.data_length["train"], "train",patch_size=args.patch_size)}


    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}

    # Get final dataloader dictionary: package using the torch-based Dataloader Class
    # dataloaders['train']: includes source-domain paired low- and high-dose data  and target-domain low-dose data 
    # dataloaders['val']: target-domain low- (for inference) and high-dose (for computing quantitative scores) data
    if args.test:
        # test using one patient data
        dataloaders = {
            'val': get_test(data_root_path=args.data_root_path,region=args.target)}

    else:
        dataloaders = {
            'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle, **kwargs),
            'val': get_test(data_root_path=args.data_root_path,region=args.target) }

    ## *********************************************************************************************************
    # initialize the model according to the name
    if args.baseline:
        if args.baseline_type == 'WGAN':
            model = WGAN(args, device=device)
        elif args.baseline_type == 'RFM':
            model = RFM(args, device=device)
        elif args.baseline_type == 'RedCNN':
            model = RedCNN(args, device=device)
        elif args.baseline_type == 'clycle':
            model = cyclegan(args, device=device)
        elif args.baseline_type == 'n2n':
            model = noiser2noise(args, device=device)
        elif args.baseline_type == 'disc':
            model = disc(args, device=device)
        elif args.baseline_type == 'uda':
            model = UDA(args, device=device)
        else:
            model = Baseline_deter(args, device=device)
    else:
        # our model
        model = DA_Denoiser(args,device=device)
    print('init sucucess')

    model.to(device)
    # To train
    if args.mode is "train":
        psnr, ssim, vif, vis = train_model(model = model,
                dataloaders = dataloaders,
                args=args
                )
        print("Run train.py Success!")
    else:
        print("\nargs.mode is wrong!\n")
    return args.measure_path,psnr, ssim, vif, vis

if __name__ == "__main__":
    ############################
    psnr_list = []
    ssim_list = []
    vif_list = []
    vis_list = []
    # set initial parameters, please go to init.py
    parsers = InitParser()
    measure_path, psnr, ssim, vif, vis = main(parsers)
    print("Run Done\n")
    #################################################################################
    ######### save quantitative score
    #################################################################################
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    vif_list.append(vif)
    vis_list.append(vis)

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    vif_list = np.array(vif_list)
    vis_list = np.array(vis_list)

    log_file = open(measure_path + '/{}_measure_final.txt'.format('mean'), 'a')
    log_file.write(
        'p_d:{:.4f},std:{:.4f},ssim_d:{:.4f},std:{:.4f},vif_d:{:.4f},std:{:.4f},vis_d:{:.4f},std:{:.4f}\n'.format(np.mean(psnr_list), np.std(psnr_list),np.mean(ssim_list), np.std(ssim_list),np.mean(vif_list), np.std(vif_list),np.mean(vis_list), np.std(vis_list)))

    log_file.close()
