import numpy as np
from glob import glob as file_glob
import scipy.stats
import scipy
import piq
import seaborn as sns
from skimage import feature
from vif_utils import vif, vif_spatial
import SimpleITK as sitk
import numpy as np  # linear algebra
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt


# Load the scans in given folder path
def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    sliceThick = []
    '''
    try:
        # distance between slices, finds slice tkickness if not availabe
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    print(slice_thickness)
    '''
    for s in slices:
        sliceThick.append(s.SliceThickness)


    return slices,sliceThick


def remove_padding(slices):
    # read the dicom images, remove padding, create 4D matrix

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0

    image[image == padding] = 0

    return np.array(image, dtype=np.int16)


def get_pixels_hu(slices):
    # read the dicom images, find HU numbers (padding, intercept, rescale), and make a 4-D array,

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0

    image[image == padding] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


#  map array between zero and 1 , find max and min
def map_0_1(array):
    out = np.zeros(array.shape)
    #    max_out = np.zeros(array.shape[0])
    #    min_out = np.zeros(array.shape[0])

    for n, val in enumerate(array):
        out[n] = (val - val.min()) / (val.max() - val.min())
    #        max_out[n] = val.max()
    #        min_out[n] = val.min()

    out = np.nan_to_num(out)

    return out.astype(np.float32)  # ,max_out,min_out


# write a nmpy array in a dicom image
def write_dicom(slices, arrays, path):
    # array should be between 0-4095
    for i in range(arrays.shape[0]):
        new_slice = slices
        pixel_array = ((arrays[i, :, :, 0] + new_slice.RescaleIntercept) / new_slice.RescaleSlope).astype(np.int16)
        # pixel_array = arrays[i,:,:,0].astype(np.int16)
        new_slice.PixelData = pixel_array.tostring()
        new_slice.save_as(path + '/' + str(i) + '.dcm')


# To have similar thickness when using different datasets
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


# This function streches the gray scale range between min and max bound for better visualization of the details
def windowing1(image, min_bound=-1000, max_bound=1000):
    output = (image - min_bound) / (max_bound - min_bound)
    output[output < 0] = 0
    output[output > 1] = 1
    return output


def windowing2(image, center, width):
    min_bound = center - width / 2
    max_bound = center + width / 2
    output = (image - min_bound) / (max_bound - min_bound)
    output[output < 0] = 0
    output[output > 1] = 1
    return output


def extract_patches(image, patch_size=32, stride=32):
    images_num, h, w = image.shape
    out = np.empty((0, patch_size, patch_size))
    sz = image.itemsize
    shape = ((h - patch_size) // stride + 1, (w - patch_size) // stride + 1, patch_size, patch_size)
    strides = sz * np.array([w * stride, stride, w, 1])

    for d in range(0, images_num):
        patches = np.lib.stride_tricks.as_strided(image[d, :, :], shape=shape, strides=strides)
        blocks = patches.reshape(-1, patch_size, patch_size)
        out = np.concatenate((out, blocks[:, :, :]))
        print(d)

    return out[:, :, :]


def normalization(image):
    mean_image = np.mean(image, axis=0).astype(np.float32)
    std_image = np.std(image, axis=0).astype(np.float32)
    out = ((image - mean_image) / std_image).astype(np.float32)
    out = np.nan_to_num(out)
    return (out, mean_image, std_image)




data_root_path = "/hdd/ckc/CT/aapm"
import matplotlib.pyplot as plt


# input_data = [[1, 0, 1],
#     [0, 2, 1],
#     [1, 1, 0]]



weights_data = [
    [[1, 0, 1],
     [-1, 1, 0],
     [0, -1, 0]],
    [[-1, 0, 1],
     [0, 0, 1],
     [1, 1, 1]]
]


def normalization(x, region='head'):
    if region == 'ABD':
        lower = -160.0
        upper = 240.0
    else:
        lower = -1000
        upper = 400


    x = (x - 1024.0 - lower) / (upper - lower)
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0
    return x

def normalization_mhd(x, region=None):
    if region == 'ABD':
        lower = -160.0
        upper = 240.0
    else:
        lower = -1000
        upper = 400

    x = (x - lower) / ((upper - lower)+0.0001)
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0
    return x

data =sitk.ReadImage('/hdd/ckc/CT/Dataset/ISICDM/high/12334826.mhd')
scan = sitk.GetArrayFromImage(data)


## Read Npy Files for LDCT images
def read_npy_LDCT(path, region='ABD'):
    temp = np.load(path).astype('float32')
    temp = normalization(temp, region)
    temp = temp[1, :, :]
    cur_img = temp.reshape([512, 512])
    return cur_img


## Read Npy Files for NDCT images
def read_npy_NDCT(path, region='ABD'):
    cur_img = np.load(path).astype('float32')
    cur_img = np.load(path).astype('float32').reshape([512, 512])
    cur_img = normalization(cur_img, region)
    return cur_img

def read_dcm_NDCT(path, region='ABD'):
    cur_img = np.load(path).astype('float32')
    cur_img = np.load(path).astype('float32').reshape([512, 512])
    cur_img = normalization(cur_img, region)
    return cur_img

# fm:[h,w]
# kernel:[k,k]
# return rs:[h,w]
def compute_conv(fm, kernel):
    [h, w] = fm.shape
    [k, _] = kernel.shape
    r = int(k / 2)
    # 定义边界填充0后的map
    padding_fm = np.zeros([h + 2, w + 2], np.float32)
    # 保存计算结果
    rs = np.zeros([h, w], np.float32)
    # 将输入在指定该区域赋值，即除了4个边界后，剩下的区域
    padding_fm[1:h + 1, 1:w + 1] = fm
    # 对每个点为中心的区域遍历
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            # 取出当前点为中心的k*k区域
            roi = padding_fm[i - r:i + r + 1, j - r:j + r + 1]

            # 计算当前点的卷积,对k*k个点点乘后求和
            kernel_input = kernel * padding_fm[i][j]
            rs[i - 1][j - 1] = np.max(np.abs(roi - kernel_input))
    return rs


def my_conv2d(input, weights):
    [h, w] = input.shape
    [_, _] = weights.shape
    outputs = np.zeros([h, w], np.float32)

    # 对每个feature map遍历，从而对每个feature map进行卷积

    # feature map==>[h,w]
    f_map = input
    # kernel ==>[k,k]
    w = weights
    rs = compute_conv(f_map, w)
    outputs = outputs + rs

    return outputs


def main():
    # shape=[c,h,w]
    kl_all_mlv = 0
    kl_input = 0
    for i in range(3):
        for j in range(3):
            #Full_Image_Paths = sorted(file_glob(data_root_path + '/show_NDCT_npy_4/*.npy'))
            Full_Image_Paths,_ = load_scan('/hdd/ckc/CT/Dataset/PHANTOM/full_1mm/120kV_107mAs')
            Full_Image = windowing1(get_pixels_hu(Full_Image_Paths),-160,240)

            # name = [i.split('_npy/')[-1] for i in Quarter_Image_Paths]

            #Full_Image = [read_dcm_NDCT(x,region='ABD') for x in Full_Image_Paths]

            # Quarter_Image = [read_npy_LDCT(x) for x in Quarter_Image_Paths]
            #input_data = np.squeeze(Full_Image[i])
            input_data = np.squeeze(Full_Image[168])

            input = np.array(input_data)
            input = input[200:264,200:264]

            plt.imshow(input,cmap='gray')
            plt.show()
            # shape=[in_c,k,k]
            weights = np.ones([3, 3])

            rs_ABD = my_conv2d(input, weights)

            # plt.imshow(rs_ABD,cmap='gray')
            # plt.show()
            # rs_ABD = np.reshape(input, [512 * 512,])
            rs_ABD = np.reshape(rs_ABD, [64 * 64, ])
            # print(rs_ABD)
            rs_ABD_sns = rs_ABD
            rs_ABD[rs_ABD == 0] = 0.

            input_ABD = np.reshape(input, [64 * 64, ])
            # print(rs_ABD)
            input_ABD_sns = input_ABD
            input_ABD[input_ABD == 0] = 0.

            px_abd = rs_ABD / np.sum(rs_ABD)
            px_input_abd = input_ABD / np.sum(input_ABD)

            #Quarter_Image_Paths = sorted(file_glob(data_root_path + '/show_LDCT_npy_3/*.npy'))

            # name = [i.split('_npy/')[-1] for i in Quarter_Image_Paths]

            Quarter_Image = np.zeros([1,512,512])
            Quarter_Image[0] = normalization_mhd(scan[110],region='ABD')#[read_npy_LDCT(x) for x in Quarter_Image_Paths]
            # Quarter_Image = [read_npy_LDCT(x,region='Head') for x in Quarter_Image_Paths]

            input_data = np.squeeze(Quarter_Image[j])

            input = np.array(input_data)
            input = input[200:264, 200:264]
            plt.imshow(input, cmap='gray')
            plt.show()
            # shape=[in_c,k,k]
            weights = np.ones([3, 3])

            rs_Head = my_conv2d(input, weights)

            rs_Head = np.reshape(rs_Head, [64 * 64, ])
            rs_Head_sns = rs_Head
            rs_Head[rs_Head == 0] = 0.
            py_head = rs_Head / np.sum(rs_Head)
            # KL = scipy.special.kl_div(px,py)
            input_head = np.reshape(input, [64 * 64, ])
            input_head_sns = input_head
            # print(rs_ABD)
            input_head[input_head == 0] = 0.
            py_input_head = input_head / np.sum(input_head)

            KL = scipy.stats.entropy(py_head, px_abd)

            KL2 = scipy.stats.entropy(py_input_head, px_input_abd)

            bins = np.linspace(0.01, 0.99, 40)
            sns.distplot(rs_Head_sns, bins, hist=True, kde=False, color='blue',
                         label='low')
            sns.distplot(rs_ABD_sns, bins, hist=True, kde=False, color='red', label='high')
            plt.title('MLV Map')
            plt.legend()
            plt.show()


            bins = np.linspace(0.01, 0.99, 40)
            sns.distplot(input_head_sns, bins, hist=True, kde=False, color='blue',
                         label='low')
            sns.distplot(input_ABD_sns, bins, hist=True, kde=False, color='red', label='high')
            plt.title('Original Map')
            plt.legend()
            plt.show()


            kl_all_mlv += KL
            kl_input += KL2

            print(KL)
            rs2 = my_conv2d(np.squeeze(Quarter_Image[j]), weights)
            fig_titles = ['low quality','MLV MAP of low quality','high quality', 'MLV MAP of high quality']
            plt.figure()
            f, axs = plt.subplots(figsize=(20, 20), nrows=2, ncols=2)
            axs[0][0].imshow(np.squeeze(Quarter_Image[j]), cmap=plt.cm.gray)
            # print('ckc')
            axs[0][0].set_title(fig_titles[0], fontsize=30)
            axs[0][0].axis('off')

            axs[1][0].imshow(rs2, cmap=plt.cm.gray)
            # print('ckc')
            axs[1][0].set_title(fig_titles[1], fontsize=30)
            axs[1][0].axis('off')

            axs[0][1].imshow(np.squeeze(Full_Image[168]), cmap=plt.cm.gray)
            axs[0][1].set_title(fig_titles[2], fontsize=30)
            axs[0][1].axis('off')

            # imshow
            rs = my_conv2d(np.squeeze(Full_Image[168]), weights)
            axs[1][1].imshow(rs, cmap=plt.cm.gray)
            axs[1][1].set_title(fig_titles[3], fontsize=30)
            axs[1][1].axis('off')
            plt.show()
            exit()
    print('KL Divergence for original map:{}'.format(kl_input / 9))
    print('KL Divergence for Maximum Local Variation map:{}'.format(kl_all_mlv / 9))


if __name__ == '__main__':
    main()
