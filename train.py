'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-24 14:57:23
@LastEditors: GuoYi
'''
import torch
import random
import os
import sys
import numpy as np
import time
from torch.optim import lr_scheduler
from torch import optim

from init import InitParser
from utils import check_dir, updata_model,save_file
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets_DA import BuildDataSet,get_test,BuildDataSet_n2n,BuildDataSet_disc,BuildDataSet_UDA,BuildDataSet_unpaired
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE,Baseline,DA_Denoiser,WGAN,RFM,RedCNN,cyclegan,noiser2noise,disc,UDA
from train_function import train_model
import glob
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(args):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "CUDA is not available"
    check_dir(args.loss_path)
    check_dir(args.model_path)
    check_dir(args.optimizer_path)
    check_dir(args.measure_path)
    check_dir(args.show_image)
    check_dir(args.code_path)
    save_file(args.code_path)


    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        args.seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        args.seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    if args.baseline == True and args.baseline_type == 'n2n':
        datasets = {
            "train": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, args.patch_n, args.data_root_path,
                                  args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                args.val_folder, False, None, args.data_length["test"], "val")}
    elif args.baseline == True and args.baseline_type == 'disc':
        datasets = {
            "train": BuildDataSet_disc(args.baseline, args.region, args.validate_region, args.patch_n,
                                      args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                    args.val_folder, False, None, args.data_length["test"], "val")}
    elif args.baseline == True and args.baseline_type == 'uda':
        datasets = {
            "train": BuildDataSet_UDA(args.baseline, args.region, args.validate_region, args.patch_n,
                                      args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                    args.val_folder, False, None, args.data_length["test"], "val")}
    else:
        if args.unpaired == True:
            datasets = {
                "train": BuildDataSet_unpaired(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
                "val": BuildDataSet(args.baseline, args.source, args.target, 1, args.data_root_path, args.val_folder,
                                    False, None, args.data_length["test"], "val")}
        else:
            datasets = {"train": BuildDataSet(args.baseline,args.source,args.target, args.patch_n,args.data_root_path, args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
                "val": BuildDataSet(args.baseline,args.source,args.target, 1,args.data_root_path, args.val_folder, False, None, args.data_length["test"], "val")}

    data_length = {x:len(datasets[x]) for x in ["train", "val"]}
    print("Data length:Train:{} Val:{}".format(data_length["train"], data_length["val"]))
    
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    # we sample patch_n patches from a image, so the batch size of the datalader
    # is equal to the total patch sizes / the patch_n
    #dataloaders = {'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle, **kwargs),'val': DataLoader(datasets['val'], args.batch_size_dataloader['val'], shuffle=False, **kwargs)}
    if args.test:
        dataloaders = {
            'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle,
                                **kwargs),
            'val': DataLoader(datasets['val'], args.batch_size_dataloader['test'], shuffle=False,
                                **kwargs)}

    else:
        # validation using one patient data
        dataloaders = {
            'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle, **kwargs),
            'val': get_test(data_root_path=args.data_root_path,region=args.target) }

    ## *********************************************************************************************************
    #print(args.baseline)
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
            model = Baseline(args, device=device)
    else:

        model = DA_Denoiser(args,device=device)
    print('init sucucess')
    #exit()
    #model = Baseline(args,device=device)
    #print(model.state_dict().keys())
    #exit()
    #exit()
    model.to(device)
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
    xt_lambda_list = [1]
    bayesian_coral_list = [10]#1e-5
    style_list = [100]#[1,10,0.1] #0.1,0.01,1,10,100,0.001
    xs_lambda_list = [1]#[0.1,0.2,0.3,0.4,0.5]
    mope_delta_list = [0.1]
    seed_lists = [0] #1
    psnr_list = []
    ssim_list = []
    vif_list = []
    vis_list = []
    for xt_lambda in xt_lambda_list:
        for bayesian_coral in bayesian_coral_list:
            for posterior_rho_init in xs_lambda_list:
                for style in style_list:
                    for seed in seed_lists:
                        for mope in mope_delta_list:
                            parsers = InitParser(xt_lambda,bayesian_coral,posterior_rho_init,style,mope,seed)
                            measure_path,psnr, ssim, vif, vis = main(parsers)
                            print("Run Done\n")
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
'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-24 14:57:23
@LastEditors: GuoYi
'''
import torch
import random
import os
import sys
import numpy as np
import time
from torch.optim import lr_scheduler
from torch import optim

from init import InitParser
from utils import check_dir, updata_model,save_file
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets_DA import BuildDataSet,get_test,BuildDataSet_n2n,BuildDataSet_disc,BuildDataSet_UDA,BuildDataSet_unpaired
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE,Baseline,DA_Denoiser,WGAN,RFM,RedCNN,cyclegan,noiser2noise,disc,UDA
from train_function import train_model
import glob
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(args):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "CUDA is not available"
    check_dir(args.loss_path)
    check_dir(args.model_path)
    check_dir(args.optimizer_path)
    check_dir(args.measure_path)
    check_dir(args.show_image)
    check_dir(args.code_path)
    save_file(args.code_path)


    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        args.seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        args.seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    if args.baseline == True and args.baseline_type == 'n2n':
        datasets = {
            "train": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, args.patch_n, args.data_root_path,
                                  args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                args.val_folder, False, None, args.data_length["test"], "val")}
    elif args.baseline == True and args.baseline_type == 'disc':
        datasets = {
            "train": BuildDataSet_disc(args.baseline, args.region, args.validate_region, args.patch_n,
                                      args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet_n2n(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                    args.val_folder, False, None, args.data_length["test"], "val")}
    elif args.baseline == True and args.baseline_type == 'uda':
        datasets = {
            "train": BuildDataSet_UDA(args.baseline, args.region, args.validate_region, args.patch_n,
                                      args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet(args.baseline, args.region, args.validate_region, 1, args.data_root_path,
                                    args.val_folder, False, None, args.data_length["test"], "val")}
    else:
        if args.unpaired == True:
            datasets = {
                "train": BuildDataSet_unpaired(args.baseline, args.source, args.target, args.patch_n, args.data_root_path,
                                      args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
                "val": BuildDataSet(args.baseline, args.source, args.target, 1, args.data_root_path, args.val_folder,
                                    False, None, args.data_length["test"], "val")}
        else:
            datasets = {"train": BuildDataSet(args.baseline,args.source,args.target, args.patch_n,args.data_root_path, args.train_folder, True, pre_trans_img, args.data_length["train"], "train"),
                "val": BuildDataSet(args.baseline,args.source,args.target, 1,args.data_root_path, args.val_folder, False, None, args.data_length["test"], "val")}

    data_length = {x:len(datasets[x]) for x in ["train", "val"]}
    print("Data length:Train:{} Val:{}".format(data_length["train"], data_length["val"]))

    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    # we sample patch_n patches from a image, so the batch size of the datalader
    # is equal to the total patch sizes / the patch_n
    #dataloaders = {'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle, **kwargs),'val': DataLoader(datasets['val'], args.batch_size_dataloader['val'], shuffle=False, **kwargs)}
    if args.test:
        dataloaders = {
            'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle,
                                **kwargs),
            'val': DataLoader(datasets['val'], args.batch_size_dataloader['test'], shuffle=False,
                                **kwargs)}

    else:
        # validation using one patient data
        dataloaders = {
            'train': DataLoader(datasets['train'], args.batch_size_dataloader['train'], shuffle=args.is_shuffle, **kwargs),
            'val': get_test(data_root_path=args.data_root_path,region=args.target) }

    ## *********************************************************************************************************
    #print(args.baseline)
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
            model = Baseline(args, device=device)
    else:

        model = DA_Denoiser(args,device=device)
    print('init sucucess')
    #exit()
    #model = Baseline(args,device=device)
    #print(model.state_dict().keys())
    #exit()
    #exit()
    model.to(device)
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
    xt_lambda_list = [0]
    bayesian_coral_list = [10]#1e-5
    style_list = [100]#[1,10,0.1] #0.1,0.01,1,10,100,0.001
    xs_lambda_list = [2]#[0.1,0.2,0.3,0.4,0.5]
    mope_delta_list = [0.1]
    seed_lists = [0] #1
    psnr_list = []
    ssim_list = []
    vif_list = []
    vis_list = []
    for xt_lambda in xt_lambda_list:
        for bayesian_coral in bayesian_coral_list:
            for posterior_rho_init in xs_lambda_list:
                for style in style_list:
                    for seed in seed_lists:
                        for mope in mope_delta_list:
                            parsers = InitParser(xt_lambda,bayesian_coral,posterior_rho_init,style,mope,seed)
                            measure_path,psnr, ssim, vif, vis = main(parsers)
                            print("Run Done\n")
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
