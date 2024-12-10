'''
@Description:
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-13 09:45:03
@LastEditors: GuoYi
'''
import torch
# import astra
import copy
import os
from glob import glob as file_glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from datasets_function import RandomCrop, ToTensor, Normalize, Scale2Gen
from datasets_function import *


## Basic datasets
##***********************************************************************************************************
class BasicData_unpaired(Dataset):
    def __init__(self, source_region,target_region, patch_n, data_root_path, Training_set, data_length, Dataset_name,folder):
        # folder means the training set
        self.folder = folder
        self.Dataset_name = Dataset_name
        self.training = Training_set
        # The data size of each epoch
        self.data_length = data_length
        self.data_length_source = 0
        self.data_length_target = 0

        if Training_set:
            print('-'*20,'...Load Training Set...region:{}'.format(source_region),'-'*20)
            Full_Image_Paths = []
            Quarter_Image_Paths = []
            # get source domain data via the selected patient id in the folder
            if source_region == 'AAPM':
                for patient in self.folder:
                    Full_Image_Paths.extend(sorted(file_glob(data_root_path + '/AAPM/full_1mm/{}/full_1mm/*.npy'.format(patient))))
                    Quarter_Image_Paths.extend(sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/{}/quarter_1mm/*.npy'.format(patient))))
            elif source_region == 'PH':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy')))

                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy')))

            elif source_region == 'ISICDM':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))

                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))


            if target_region == 'PH':
                Quarter_Image_Paths_T = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
            elif target_region == 'ISICDM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/ISICDM/low/*.npy'))
                Quarter_Image_Paths_T_GT = sorted(
                    file_glob(data_root_path + '/ISICDM/high/*.npy'))
            elif target_region == 'AAPM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))

                Quarter_Image_Paths_T_GT = sorted(
                    file_glob(data_root_path + '/ISICDM/high/*.npy'))

            ### Note the AAPM and PHANTON data have been transformed into the HU range
            ### Note the isicdm data have HU format
            self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Full_Image_Paths])
            self.Quarter_Image = np.vstack([clip_map_0_1_hu(x,min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths])
            self.data_length_source = len(self.Full_Image)
            self.Quarter_Image_T = np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths_T])
            self.data_length_target = len(self.Quarter_Image_T)
            self.Quarter_Image_T_GT = np.vstack(
                [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths_T_GT])
            self.data_length_target_GT = len(self.Quarter_Image_T_GT)

            print(self.Quarter_Image_T.shape)


        else:
            if self.Dataset_name == 'test':
                print('-' * 20, '...Load Testing Set...region:{}'.format(target_region), '-' * 20)
                Full_Image_Paths = []
                Quarter_Image_Paths = []
                if target_region == 'PH':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
                elif target_region == 'ISICDM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/high/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/*.npy'))

                elif target_region == 'AAPM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))


                self.Full_Image =  np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Full_Image_Paths])
                self.Quarter_Image = np.vstack([clip_map_0_1_hu(x,min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths])
                self.data_length_source = len(self.Full_Image)


            else:
                print('-' * 20, '...Load Testing Set...region:{}'.format(target_region), '-' * 20)
                Full_Image_Paths = []
                Quarter_Image_Paths = []
                if target_region == 'PH':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
                elif target_region == 'ISICDM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/high/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/*.npy'))
                elif target_region == 'AAPM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))

                self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
                self.Quarter_Image = np.vstack(
                    [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])
                self.data_length_source = len(self.Full_Image)
    def __len__(self):
        return self.data_length_source

    def __getitem__(self, idx):
        # patient_index = np.random.randint(len(self.folder)) ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image
        quarter_image_set = self.Quarter_Image
        if self.training:
            quarter_image_set_T = self.Quarter_Image_T
            quarter_image_set_T_GT = self.Quarter_Image_T_GT
        image_index = np.random.randint(0,self.data_length_source)  ## Three consecutive images were randomly selected from people with the disease
        if self.training:
            image_index_T = np.random.randint(0,self.data_length_target)
            image_index_T_GT = np.random.randint(0,self.data_length_target_GT)
        # print(image_index)
        if self.Dataset_name is "test":
            image_index = 10

        full_image = full_image_set[image_index]
        quarter_image = quarter_image_set[image_index]
        if self.training:
            quarter_image_T = quarter_image_set_T[image_index_T]
            quarter_image_T_GT = quarter_image_set_T_GT[image_index_T_GT]
            return full_image, quarter_image,quarter_image_T,quarter_image_T_GT
        else:
            return full_image, quarter_image



class BasicData(Dataset):
    def __init__(self, source_region,target_region, patch_n, data_root_path, Training_set, data_length, Dataset_name,folder):
        # folder means the training set
        self.folder = folder
        self.Dataset_name = Dataset_name
        self.training = Training_set
        self.data_length = data_length  ## The data size of each epoch
        self.data_length_source = 0
        self.data_length_target = 0

        if Training_set:
            print('-'*20,'...Load Training Set...region:{}'.format(source_region),'-'*20)
            Full_Image_Paths = []
            Quarter_Image_Paths = []
            # get source domain data via the selected patient id in the folder
            if source_region == 'AAPM':
                for patient in self.folder:
                    Full_Image_Paths.extend(sorted(file_glob(data_root_path + '/AAPM/full_1mm/{}/full_1mm/*.npy'.format(patient))))
                    Quarter_Image_Paths.extend(sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/{}/quarter_1mm/*.npy'.format(patient))))
            elif source_region == 'PH':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy')))

                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy')))

            elif source_region == 'ISICDM':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))

                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))


            if target_region == 'PH':
                Quarter_Image_Paths_T = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
            elif target_region == 'ISICDM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/ISICDM/low/*.npy'))

            elif target_region == 'AAPM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
            elif target_region == "AAPM_5":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/120000_1mm/*/120000_1mm/*.npy'))
            elif target_region == "AAPM_50":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/1100000_1mm/*/1100000_1mm/*.npy'))

            ### Note the AAPM and PHANTON data have been transformed into the HU range
            ### Note the isicdm data have HU format
            self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Full_Image_Paths])
            self.Quarter_Image = np.vstack([clip_map_0_1_hu(x,min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths])
            self.data_length_source = len(self.Full_Image)
            self.Quarter_Image_T = np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths_T])
            self.data_length_target = len(self.Quarter_Image_T)


        else:
            if self.Dataset_name == 'test':
                print('-' * 20, '...Load Testing Set...region:{}'.format(target_region), '-' * 20)
                Full_Image_Paths = []
                Quarter_Image_Paths = []
                if target_region == 'PH':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
                elif target_region == 'ISICDM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/high/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/*.npy'))

                elif target_region == 'AAPM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
                elif target_region == "AAPM_5":
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
                elif target_region == "AAPM_50":
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))

                self.Full_Image =  np.vstack([clip_map_0_1_hu(x, min_hu=-1000,max_hu=400) for x in Full_Image_Paths])
                self.Quarter_Image = np.vstack([clip_map_0_1_hu(x,min_hu=-1000,max_hu=400) for x in Quarter_Image_Paths])
                self.data_length_source = len(self.Full_Image)


            else:
                print('-' * 20, '...Load Testing Set...region:{}'.format(target_region), '-' * 20)
                Full_Image_Paths = []
                Quarter_Image_Paths = []
                if target_region == 'PH':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
                elif target_region == 'ISICDM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/high/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/*.npy'))
                elif target_region == 'AAPM':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
                elif target_region == "AAPM_5":
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
                elif target_region == "AAPM_50":
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))

                self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
                self.Quarter_Image = np.vstack(
                    [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])
                self.data_length_source = len(self.Full_Image)
    def __len__(self):
        return self.data_length_source

    def __getitem__(self, idx):
        # patient_index = np.random.randint(len(self.folder))                     ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image
        quarter_image_set = self.Quarter_Image
        if self.training:
            quarter_image_set_T = self.Quarter_Image_T
        image_index = np.random.randint(0,self.data_length_source)  ## Three consecutive images were randomly selected from people with the disease
        if self.training:
            image_index_T = np.random.randint(0,self.data_length_target)

        # print(image_index)
        if self.Dataset_name is "test":
            image_index = 10

        full_image = full_image_set[image_index]
        quarter_image = quarter_image_set[image_index]
        if self.training:
            quarter_image_T = quarter_image_set_T[image_index_T]

            return full_image, quarter_image,quarter_image_T
        else:
            return full_image, quarter_image
def get_patch(full_input_img, full_target_img, patch_n, patch_size, drop_background=0.1):  # 0.1
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    c, h, w = full_input_img.shape

    new_h, new_w = patch_size, patch_size
    n = 0
    continue_label = 0

    if patch_size == 512:
        patch_input_imgs.append(full_input_img[:, :, :])
        patch_target_imgs.append(full_target_img[:, :,:])
    else:
        while n < patch_n:
            # if the patch covers the whole image
            if patch_size == full_input_img.shape[1]:
                top = 0
                left = 0
            else:
                if continue_label == 0:
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                else:
                    top = 223 + np.random.randint(0, 32)
                    left = 223 + np.random.randint(0, 32)


            patch_input_img = full_input_img[:, top:top + new_h, left:left + new_w]
            patch_target_img = full_target_img[:, top:top + new_h, left:left + new_w]
            if (np.mean(patch_input_img) < drop_background):
                continue_label += 1
                continue
            else:
                n += 1
                patch_input_imgs.append(patch_input_img)
                patch_target_imgs.append(patch_target_img)


    return np.array(patch_input_imgs), np.array(patch_target_imgs)
def get_patch_label(full_input_img, full_target_img, patch_n, patch_size, drop_background=0.1,source=True):  # 0.1
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    c, h, w = full_input_img.shape

    new_h, new_w = patch_size, patch_size
    n = 0
    continue_label = 0

    while n < patch_n:
        # if the patch covers the whole image
        if patch_size == full_input_img.shape[1]:
            top = 0
            left = 0
        else:
            if continue_label == 0:
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
            else:
                top = 317 + np.random.randint(0, 32)
                left = 252 + np.random.randint(0, 32)


        patch_input_img = full_input_img[:, top:top + new_h, left:left + new_w]
        if source:
            patch_target_img = np.zeros([1,6,6])
        else:
            patch_target_img = np.ones([1,6,6])
        # print(np.mean(patch_input_img))
        if (np.mean(patch_input_img) < drop_background):
            continue_label += 1
            continue
        else:
            n += 1
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)

    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_patch_label_noiser(full_input_img, full_target_img, patch_n, patch_size, drop_background=0.1,source=True):  # 0.1
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    patch_target_labels = []
    c, h, w = full_input_img.shape

    new_h, new_w = patch_size, patch_size
    n = 0
    continue_label = 0
    while n < patch_n:
        # if the patch covers the whole image
        if patch_size == full_input_img.shape[1]:
            top = 0
            left = 0
        else:
            if continue_label == 0:
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
            else:
                top = 317 + np.random.randint(0, 32)
                left = 252 + np.random.randint(0, 32)


        patch_input_img = full_input_img[:, top:top + new_h, left:left + new_w]
        patch_target_img = full_target_img[:, top:top + new_h, left:left + new_w]
        if source:
            patch_target_label = np.zeros([1,6,6])
        else:
            patch_target_label = np.ones([1,6,6])

        if (np.mean(patch_input_img) < drop_background):
            continue_label += 1
            continue
        else:
            n += 1
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
            patch_target_labels.append(patch_target_label)
    return np.array(patch_input_imgs), np.array(patch_target_imgs), np.array(patch_target_labels)

def get_patch_target(full_input_img, patch_n, patch_size, drop_background=0.1):  # 0.1

    patch_input_imgs = []

    c, h, w = full_input_img.shape

    new_h, new_w = patch_size, patch_size
    n = 0
    continue_label = 0
    while n < patch_n:
        # if the patch covers the whole image
        if patch_size == full_input_img.shape[1]:
            top = 0
            left = 0
        else:
            if continue_label == 0:
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
            else:
                top = 223 + np.random.randint(0, 32)
                left = 223 + np.random.randint(0, 32)


        patch_input_img = full_input_img[:, top:top + new_h, left:left + new_w]


        if (np.mean(patch_input_img) < drop_background):
            continue_label += 1
            continue
        else:
            n += 1
            patch_input_imgs.append(patch_input_img)


    return np.array(patch_input_imgs)

def get_test(data_root_path, region='abd',show=False):


    if show:
        Full_Image_Paths = []
        Quarter_Image_Paths = []
        if region == 'PH':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/120kV_42mAs/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/120kV_42mAs/*.npy'))
        elif region == 'ISICDM':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
        elif region == 'AAPM':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
        elif region == "AAPM_5":
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
        elif region == "AAPM_50":
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))

    else:
        Full_Image_Paths = []
        Quarter_Image_Paths = []
        if region == 'PH':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/120kV_42mAs/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/120kV_42mAs/*.npy'))
        elif region == 'ISICDM':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/high_roi/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low_roi/*.npy'))
        elif region == 'AAPM':
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
        elif region == "AAPM_5":
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
        elif region == "AAPM_50":
            Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
            Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))

    Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
    Quarter_Image = np.vstack(
        [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])

    data = {'full_image': Full_Image, 'quarter_image': Quarter_Image}

    return data





def get_test_n2n(data_root_path, region='abd',show=False,noise=15):
    noise_intensity = noise/255.
    Full_Image_Paths = []
    Quarter_Image_Paths = []
    if region == 'PH':
        Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/120kV_42mAs/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/120kV_42mAs/*.npy'))
    elif region == 'ISICDM':
        Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
    elif region == 'AAPM':
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
    elif region == "AAPM_5":
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
    elif region == "AAPM_50":
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))


    Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
    Quarter_Image = np.vstack(
    [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])
    Quarter_Image_noiser = []
    for x in range(Quarter_Image.shape[0]):
        quarter_image_noiser = Quarter_Image[x] + np.random.randn(*Quarter_Image[x].shape) * noise_intensity
        Quarter_Image_noiser.append(quarter_image_noiser)
    Quarter_Image_noiser = np.stack(Quarter_Image_noiser)
    data = {'full_image': Full_Image, 'quarter_image': Quarter_Image_noiser}

    return data

def get_test_uda(data_root_path, region='abd',show=False):
    Full_Image_Paths = []
    Quarter_Image_Paths = []
    if region == 'PH':
        Full_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/120kV_42mAs/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/120kV_42mAs/*.npy'))
    elif region == 'ISICDM':

        Full_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/ISICDM/low/12399485.npy'))
    elif region == 'AAPM':
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
    elif region == "AAPM_5":
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/120000_1mm/L506/120000_1mm/*.npy'))
    elif region == "AAPM_50":
        Full_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/full_1mm/L506/full_1mm/*.npy'))
        Quarter_Image_Paths = sorted(file_glob(data_root_path + '/AAPM/1100000_1mm/L506/1100000_1mm/*.npy'))

    Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
    Quarter_Image = np.vstack(
        [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])

    data = {'full_image': Full_Image, 'quarter_image': Quarter_Image}

    return data

def get_test_patch(data_root_path, region='abd'):
    Full_Image_Paths = sorted(file_glob(data_root_path + '/{}/show_NDCT_npy/*.npy'.format(region)))
    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/{}/show_LDCT_npy/*.npy'.format(region)))
    Full_Image = np.stack([read_npy_NDCT(x, region=region) for x in Full_Image_Paths])
    Quarter_Image = np.stack([read_npy_LDCT(x, region=region) for x in Quarter_Image_Paths])

    total = Full_Image.shape[0]
    output = []
    for i in range(total):
        output.append(get_patch_target(Quarter_Image[i],patch_n=8,patch_size=64))
    output = np.concatenate(output,axis=0)
    return output



class BuildDataSet(Dataset):
    def __init__(self, baseline,source_region,target_region, patch_n, data_root_path, folder, training_set=True, pre_trans_img=None, data_length=None,
                 Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.patch_n = patch_n
        self.pre_trans_img = pre_trans_img
        self.baseline = baseline
        self.training_set = training_set
        self.imgset = BasicData(source_region,target_region, patch_n, data_root_path, training_set, data_length,
                                Dataset_name=self.Dataset_name,folder=folder)
        if Dataset_name == 'train':
            self.patch_size = patch_size
        else:
            self.patch_size = 512

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):
        new_image = Transf(image[0])

        return new_image

    def __getitem__(self, idx):
        if self.training_set:
            full_image, quarter_image,quarter_image_T = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)
            if not self.baseline:
                quarter_patches_T = get_patch_target(quarter_image_T, patch_n=self.patch_n,
                                                          patch_size=self.patch_size)
                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches,
                          "quarter_image_target": quarter_patches_T}
            else:
                quarter_patches_T = get_patch_target(quarter_image_T, patch_n=self.patch_n,
                                                     patch_size=self.patch_size)
                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches,
                          "quarter_image_target": quarter_patches_T}
            return sample
        else:

            full_image, quarter_image = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)

            sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches}
            return sample



class BuildDataSet_unpaired(Dataset):
    def __init__(self, baseline,source_region,target_region, patch_n, data_root_path, folder, training_set=True, pre_trans_img=None, data_length=None,
                 Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.patch_n = patch_n
        self.pre_trans_img = pre_trans_img
        self.baseline = baseline
        self.training_set = training_set
        self.imgset = BasicData_unpaired(source_region,target_region, patch_n, data_root_path, training_set, data_length,
                                Dataset_name=self.Dataset_name,folder=folder)
        if Dataset_name == 'train':
            self.patch_size = 256
        else:
            self.patch_size = 512

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):
        new_image = Transf(image[0])
        return new_image

    def __getitem__(self, idx):
        if self.training_set:
            full_image, quarter_image,quarter_image_T,quarter_image_T_GT = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)
            if not self.baseline:
                quarter_patches_T = get_patch_target(quarter_image_T, patch_n=self.patch_n,
                                                          patch_size=self.patch_size)
                quarter_patches_T_GT = get_patch_target(quarter_image_T_GT, patch_n=self.patch_n,
                                                     patch_size=self.patch_size)
                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches,
                          "quarter_image_target": quarter_patches_T,
                          "quarter_image_target_GT": quarter_patches_T_GT}
            else:
                quarter_patches_T = get_patch_target(quarter_image_T, patch_n=self.patch_n,
                                                     patch_size=self.patch_size)
                quarter_patches_T_GT = get_patch_target(quarter_image_T_GT, patch_n=self.patch_n,
                                                        patch_size=self.patch_size)
                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches,
                          "quarter_image_target": quarter_patches_T, "quarter_image_target_GT": quarter_patches_T_GT}
            return sample
        else:

            full_image, quarter_image = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)

            sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches}
            return sample

class BuildDataSet_n2n(Dataset):
    def __init__(self, baseline,source,target_region, patch_n, data_root_path, folder, training_set=True, pre_trans_img=None, data_length=None,
                 Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.patch_n = patch_n
        self.pre_trans_img = pre_trans_img
        self.baseline = baseline
        self.training_set = training_set
        self.imgset = BasicData_n2n(target_region, patch_n, data_root_path, training_set, data_length,
                                Dataset_name=self.Dataset_name)
        if Dataset_name == 'train':
            self.patch_size = 256
        else:
            self.patch_size = 512

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):

        new_image = Transf(image[0])
        return new_image

    def __getitem__(self, idx):
        if self.training_set:
            full_image, quarter_image = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)


            sample = {"full_image": full_patches,
                      "quarter_image": quarter_patches}
            return sample

## Basic datasets
##***********************************************************************************************************
class BasicData_n2n(Dataset):
    def __init__(self, target_region, patch_n, data_root_path, folder, data_length, Dataset_name):
        # folder means the training set
        self.folder = folder
        self.Dataset_name = Dataset_name
        self.noise_intensity = 15/255.
        self.data_length = data_length  ## The data size of each epoch
        if folder:
            print('-'*20,'...Load Training Set...region:{}'.format(target_region),'-'*20)

            # get source domain data via the selected patient id in the folder
            if target_region == 'PH':
                Quarter_Image_Paths_T = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
            elif target_region == 'ISICDM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/ISICDM/low/*.npy'))

            elif target_region == 'AAPM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
            elif target_region == "AAPM_5":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/120000_1mm/*/120000_1mm/*.npy'))
            elif target_region == "AAPM_50":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/1100000_1mm/*/1100000_1mm/*.npy'))

            ### Note the AAPM and PHANTON data have been transformed into the HU range
            ### Note the isicdm data have HU format
            self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths_T])

            self.data_length_source = len(self.Full_Image)

            self.data_length_target = len(self.Full_Image)

            self.Quarter_Image = []
            for x in range(self.Full_Image.shape[0]):
                quarter_image_noiser = self.Full_Image[x] + np.random.randn(*self.Full_Image[x].shape) * self.noise_intensity
                self.Quarter_Image.append(quarter_image_noiser)
            print(len(self.Full_Image))
            print(len(self.Quarter_Image))

        else:
            if self.Dataset_name == 'test':
                print('-' * 20, '...Load Test Set...region:{}'.format(target_region), '-' * 20)
                if target_region == 'abd':
                    Full_Image_Paths = sorted(
                        file_glob(data_root_path + '/{}/test_NDCT_npy/*.npy'.format(target_region)))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/{}/test_LDCT_npy/*.npy'.format(target_region)))
                else:
                    Full_Image_Paths = sorted(
                        file_glob(data_root_path + '/{}/test_NDCT_npy/*.npy'.format(target_region)))
                    Quarter_Image_Paths = sorted(
                        file_glob(data_root_path + '/{}/test_LDCT_npy/*.npy'.format(target_region)))

                self.Full_Image = [read_npy_NDCT(x, region=target_region) for x in Full_Image_Paths]

                self.Quarter_Image = []
                for x in Quarter_Image_Paths:
                    quarter_image = read_npy_LDCT(x, region=target_region)
                    quarter_image_noiser = quarter_image + np.random.randn(*quarter_image.shape) * self.noise_intensity
                    self.Quarter_Image.append(quarter_image_noiser)


            elif self.Dataset_name == 'show':
                Full_Image_Paths = sorted(
                    file_glob(data_root_path + '/{}/show_NDCT_npy/*.npy'.format(target_region)))
                Quarter_Image_Paths = sorted(
                    file_glob(data_root_path + '/{}/show_LDCT_npy/*.npy'.format(target_region)))
                self.Full_Image = [read_npy_NDCT(x, region=target_region) for x in Full_Image_Paths]
                # print(len(self.Full_Image))
                self.Quarter_Image = []
                for x in Quarter_Image_Paths:
                    quarter_image = read_npy_LDCT(x, region=target_region)
                    quarter_image_noiser = quarter_image + np.random.randn(*quarter_image.shape) * self.noise_intensity
                    self.Quarter_Image.append(quarter_image_noiser)
            else:
                print('-' * 20, '...Load Validate Set...region:{}'.format(target_region), '-' * 20)
                if target_region == 'abd':
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_NDCT_npy/*.npy'.format(target_region)))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_LDCT_npy/*.npy'.format(target_region)))
                else:
                    Full_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_NDCT_npy/*.npy'.format(target_region)))
                    Quarter_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_LDCT_npy/*.npy'.format(target_region)))

                self.Full_Image = [read_npy_NDCT(x, region=target_region) for x in Full_Image_Paths]

                self.Quarter_Image = []
                for x in Quarter_Image_Paths:
                    quarter_image = read_npy_LDCT(x, region=target_region)
                    quarter_image_noiser = quarter_image + np.random.randn(*quarter_image.shape) * self.noise_intensity
                    self.Quarter_Image.append(quarter_image_noiser)



    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image
        quarter_image_set = self.Quarter_Image

        image_index = np.random.randint(0,
                                        len(full_image_set))  ## Three consecutive images were randomly selected from people with the disease

        #
        if self.Dataset_name is "test":
            image_index = 10
        full_image = full_image_set[image_index]
        quarter_image = quarter_image_set[image_index]

        return full_image, quarter_image




class BuildDataSet_disc(Dataset):
    def __init__(self, baseline,source_region,target_region, patch_n, data_root_path, folder, training_set=True, pre_trans_img=None, data_length=None,
                 Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.patch_n = patch_n
        self.pre_trans_img = pre_trans_img
        self.baseline = baseline
        self.training_set = training_set
        self.imgset = BasicData_disc(source_region,target_region, patch_n, data_root_path, training_set, data_length,
                                Dataset_name=self.Dataset_name,folder=folder)
        if Dataset_name == 'train':
            self.patch_size = 64
        else:
            self.patch_size = 512

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):

        new_image = Transf(image[0])

        return new_image

    def __getitem__(self, idx):
        if self.training_set:
            full_image, quarter_image, quarter_image_T = self.imgset[idx]
            quarter_patches_S, full_patches_S = get_patch_label(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size,source=True)
            label_source = []
            if not self.baseline:
                quarter_patches_T,full_patches_T = get_patch_label(quarter_image_T, quarter_image_T,patch_n=self.patch_n,
                                                     patch_size=self.patch_size,source=False)
                sample = {"full_image": full_patches_S,
                          "quarter_image": quarter_patches_S,
                          "quarter_image_target": quarter_patches_T
                          }
            else:
                quarter_patches_T, full_patches_T = get_patch_label(quarter_image_T, quarter_image_T,
                                                                    patch_n=self.patch_n,
                                                                    patch_size=self.patch_size, source=False)

                full_patches = np.concatenate((full_patches_S,full_patches_T),axis=0)
                quarter_patches = np.concatenate((quarter_patches_S,quarter_patches_T),axis=0)

                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches}
            return sample
        else:

            full_image, quarter_image = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)

            sample = {"full_image": full_patches,
                      "quarter_image": quarter_patches}
            return sample

## Basic datasets
##***********************************************************************************************************
class BasicData_disc(Dataset):
    def __init__(self, source_region,target_region, patch_n, data_root_path, Training_set, data_length, Dataset_name,folder):
        # folder means the training set
        self.folder = folder
        self.Dataset_name = Dataset_name

        self.data_length = data_length  ## The data size of each epoch

        if folder:
            print('-'*20,'...Load Training Set...region:{}'.format(source_region),'-'*20)
            Full_Image_Paths = []
            Quarter_Image_Paths = []
            # get source domain data via the selected patient id in the folder
            if source_region == 'AAPM':
                for patient in self.folder:
                    Full_Image_Paths.extend(
                        sorted(file_glob(data_root_path + '/AAPM/full_1mm/{}/full_1mm/*.npy'.format(patient))))
                    Quarter_Image_Paths.extend(
                        sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/{}/quarter_1mm/*.npy'.format(patient))))
            elif source_region == 'PH':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy')))
                # Full_Image_Paths.remove(data_root_path+ '/PHANTOM/full_1mm/120kV_64mAs/data.npy')
                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy')))
                # Quarter_Image_Paths.remove(data_root_path + '/PHANTOM/quarter_1mm/120kV_64mAs/data.npy')
            elif source_region == 'ISICDM':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))
                # Full_Image_Paths.remove(data_root_path+ '/PHANTOM/full_1mm/120kV_64mAs/data.npy')
                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))

            if target_region == 'PH':
                Quarter_Image_Paths_T = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
            elif target_region == 'ISICDM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/ISICDM/low/*.npy'))

            elif target_region == 'AAPM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
            elif target_region == "AAPM_5":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/120000_1mm/*/120000_1mm/*.npy'))
            elif target_region == "AAPM_50":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/1100000_1mm/*/1100000_1mm/*.npy'))

            ### Note the AAPM and PHANTON data have been transformed into the HU range
            ### Note the isicdm data have HU format
            self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
            self.Quarter_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])
            self.data_length_source = len(self.Full_Image)
            self.Quarter_Image_T = np.vstack(
                [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths_T])
            self.data_length_target = len(self.Quarter_Image_T)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image
        quarter_image_set = self.Quarter_Image
        if self.folder:
            quarter_image_set_T = self.Quarter_Image_T

        image_index = np.random.randint(0,
                                        len(full_image_set))  ## Three consecutive images were randomly selected from people with the disease
        if self.folder:
            image_index_T = np.random.randint(0,
                                              len(quarter_image_set_T))
        #
        if self.Dataset_name is "test":
            image_index = 10
        full_image = full_image_set[image_index]
        quarter_image = quarter_image_set[image_index]
        if self.folder:
            quarter_image_T = quarter_image_set_T[image_index_T]

            return full_image, quarter_image, quarter_image_T
        else:
            return full_image, quarter_image


class BuildDataSet_UDA(Dataset):
    def __init__(self, baseline,source_region,target_region, patch_n, data_root_path, folder, training_set=True, pre_trans_img=None, data_length=None,
                 Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.patch_n = patch_n
        self.pre_trans_img = pre_trans_img
        self.baseline = baseline
        self.training_set = training_set
        self.imgset = BasicData_UDA(source_region,target_region, patch_n, data_root_path, training_set, data_length,
                                Dataset_name=self.Dataset_name,folder=folder)
        if Dataset_name == 'train':
            self.patch_size = 64
        else:
            self.patch_size = 512

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2, size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):

        new_image = Transf(image[0])
        return new_image

    def __getitem__(self, idx):
        if self.training_set:
            full_image, quarter_image, quarter_image_T,quarter_image_T_noiser = self.imgset[idx]
            quarter_patches_S, full_patches_S = get_patch_label(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size,source=False)
            label_source = []
            if not self.baseline:
                quarter_patches_T_noiser,quarter_patches_T,full_patches_T = get_patch_label(quarter_image_T_noiser, quarter_image_T,patch_n=self.patch_n,
                                                     patch_size=self.patch_size,source=False)
                sample = {"full_image": full_patches_S,
                          "quarter_image": quarter_patches_S,
                          "quarter_image_target": quarter_patches_T
                          }
            else:
                quarter_patches_T_noiser, quarter_patches_T, full_patches_T = get_patch_label_noiser(quarter_image_T_noiser,
                                                                                              quarter_image_T,
                                                                                              patch_n=self.patch_n,
                                                                                              patch_size=self.patch_size,
                                                                                              source=False)

                full_patches = full_patches_S
                quarter_patches = quarter_patches_S

                full_patches_T_target = full_patches_T
                quarter_patches_T_input = quarter_patches_T



                sample = {"full_image": full_patches,
                          "quarter_image": quarter_patches,
                          "full_image_T": full_patches_T_target,
                          "quarter_image_T": quarter_patches_T_input,"quarter_image_T_noise": quarter_patches_T_noiser}
            return sample
        else:

            full_image, quarter_image = self.imgset[idx]
            quarter_patches, full_patches = get_patch(quarter_image, full_image, patch_n=self.patch_n,
                                                      patch_size=self.patch_size)

            sample = {"full_image": full_patches,
                      "quarter_image": quarter_patches}
            return sample

## Basic datasets
##***********************************************************************************************************
class BasicData_UDA(Dataset):
    def __init__(self, source_region,target_region, patch_n, data_root_path, Training_set, data_length, Dataset_name,folder):
        # folder means the training set
        self.folder = folder
        self.Dataset_name = Dataset_name
        self.noise_intensity = 10/255.
        self.data_length = data_length  ## The data size of each epoch
        # self.Full_Image = {x:read_mat(data_root_path + "/{}_full_1mm_CT.mat".format(x)) for x in self.folder}        ## High-dose images of all patients
        if folder:
            print('-'*20,'...Load Training Set...region:{}'.format(source_region),'-'*20)
            Full_Image_Paths = []
            Quarter_Image_Paths = []
            # get source domain data via the selected patient id in the folder
            if source_region == 'AAPM':
                for patient in self.folder:
                    Full_Image_Paths.extend(
                        sorted(file_glob(data_root_path + '/AAPM/full_1mm/{}/full_1mm/*.npy'.format(patient))))
                    Quarter_Image_Paths.extend(
                        sorted(file_glob(data_root_path + '/AAPM/quarter_1mm/{}/quarter_1mm/*.npy'.format(patient))))
            elif source_region == 'PH':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/full_1mm/*/*.npy')))

                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy')))

            elif source_region == 'ISICDM':
                Full_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))
                Quarter_Image_Paths.extend(
                    sorted(file_glob(data_root_path + '/ISICDM/low/*.npy')))

            if target_region == 'PH':
                Quarter_Image_Paths_T = sorted(file_glob(data_root_path + '/PHANTOM/quarter_1mm/*/*.npy'))
            elif target_region == 'ISICDM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/ISICDM/low/*.npy'))

            elif target_region == 'AAPM':
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/quarter_1mm/L506/quarter_1mm/*.npy'))
            elif target_region == "AAPM_5":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/120000_1mm/*/120000_1mm/*.npy'))
            elif target_region == "AAPM_50":
                Quarter_Image_Paths_T = sorted(
                    file_glob(data_root_path + '/AAPM/1100000_1mm/*/1100000_1mm/*.npy'))

            ### Note the AAPM and PHANTON data have been transformed into the HU range
            ### Note the isicdm data have HU format
            self.Full_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Full_Image_Paths])
            self.Quarter_Image = np.vstack([clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths])
            self.data_length_source = len(self.Full_Image)
            self.Quarter_Image_T = np.vstack(
                [clip_map_0_1_hu(x, min_hu=-1000, max_hu=400) for x in Quarter_Image_Paths_T])
            self.Quarter_Image_T_noiser = []
            for x in  range(self.Quarter_Image_T.shape[0]):
                quarter_image_noiser = self.Quarter_Image_T[x] +  np.random.randn(*self.Quarter_Image_T[x].shape) * self.noise_intensity
                self.Quarter_Image_T_noiser.append(quarter_image_noiser)
        else:
            print('-' * 20, '...Load Validate Set...region:{}'.format(target_region), '-' * 20)
            if target_region == 'abd':
                Full_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_NDCT_npy/*.npy'.format(target_region)))
                Quarter_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_LDCT_npy/*.npy'.format(target_region)))
            else:
                Full_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_NDCT_npy/*.npy'.format(target_region)))
                Quarter_Image_Paths = sorted(file_glob(data_root_path + '/{}/validate_LDCT_npy/*.npy'.format(target_region)))

            self.Full_Image = [read_npy_NDCT(x, region=target_region) for x in Full_Image_Paths]

            self.Quarter_Image = [read_npy_LDCT(x, region=target_region) for x in Quarter_Image_Paths]



    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image
        quarter_image_set = self.Quarter_Image
        if self.folder:
            quarter_image_set_T = self.Quarter_Image_T
            quarter_image_set_T_noiser = self.Quarter_Image_T_noiser
        image_index = np.random.randint(0,
                                        len(full_image_set))  ## Three consecutive images were randomly selected from people with the disease
        if self.folder:
            image_index_T = np.random.randint(0,
                                              len(quarter_image_set_T))

        if self.Dataset_name is "test":
            image_index = 10
        full_image = full_image_set[image_index]
        quarter_image = quarter_image_set[image_index]
        if self.folder:
            quarter_image_T = quarter_image_set_T[image_index_T]
            quarter_image_T_noiser = quarter_image_set_T_noiser[image_index_T]
            return full_image, quarter_image, quarter_image_T,quarter_image_T_noiser
        else:
            return full_image, quarter_image
