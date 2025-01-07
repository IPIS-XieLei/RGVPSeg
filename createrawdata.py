import time, os
import torch
import numpy
import nibabel as nib
import numpy as np
from scipy import ndimage as nd

def cut_data(data_path, save_path, data_num):
    print('-' * 30)
    print('Begin cut data')
    print('-' * 30)

    # img_t1 = nib.load(data_path)
    # img_t1_affine = img_t1.affine
    # img_t1_data = img_t1.get_fdata()
    # img_t1_data = np.squeeze(img_t1_data)

    img_fa = nib.load(data_path)
    img_fa_affine = img_fa.affine
    img_fa_data = img_fa.get_fdata()
    img_fa_data = np.squeeze(img_fa_data)


    # T1_cut_data = img_t1_data[8:136, 7:167, 0:128]
    FA_cut_data = img_fa_data[8:136, 7:167, 0:128]

    # T1_save_path = save_path + "/" + data_num + '/new_T1.nii.gz'
    # T1_cut_data = nib.Nifti1Image(T1_cut_data, img_t1_affine)
    # nib.save(T1_cut_data, T1_save_path)

    FA_save_path = save_path + "/" + data_num + '/new_FA.nii.gz'
    FA_cut_data = nib.Nifti1Image(FA_cut_data, img_fa_affine)
    nib.save(FA_cut_data, FA_save_path)


    print('-' * 30)

def get_255(data_path,data_num):
    print('-' * 30)
    print('Begin get 255')
    print('-' * 30)
    # for each volume do:
    # img_t1_name = 'new_T1.nii.gz'
    # img_t1_name = os.path.join(data_path, img_t1_name)
    #
    # img_fa_name = 'new_FA.nii.gz'
    # img_fa_name = os.path.join(data_path, img_fa_name)

    img_t2_name = data_num+'_ON-T2.nii.gz'
    img_t2_name = os.path.join(data_path, img_t2_name)

    # img_t1 = nib.load(img_t1_name)
    # img_t1_affine = img_t1.affine
    # img_t1_data = img_t1.get_fdata()
    # img_t1_data = np.squeeze(img_t1_data)
    #
    # img_fa = nib.load(img_fa_name)
    # img_fa_data = img_fa.get_fdata()
    # img_fa_affine = img_fa.affine
    # img_fa_data = np.squeeze(img_fa_data)
    # img_fa_data = np.nan_to_num(img_fa_data)

    img_t2 = nib.load(img_t2_name)
    img_t2_affine = img_t2.affine
    img_t2_data = img_t2.get_fdata()
    img_t2_data = np.squeeze(img_t2_data)

    # min = np.nanmin(img_fa_data[:, :, 20])
    # max = np.nanmax(img_fa_data[:, :, 20])

    X = img_t2_data.shape
    # empty_t1 = np.zeros([128, 160, 128])
    # empty_fa = np.zeros([128, 160, 128])
    empty_t2 = np.zeros([128, 160, 128])
    for slice in range(X[2]):
        # array_t1 = 255 * (img_t1_data[:, :, slice] - np.min(img_t1_data[:, :, slice])) / (
        #             np.max(img_t1_data[:, :, slice]) - np.min(img_t1_data[:, :, slice]))
        # array_fa = 255 * (img_fa_data[:, :, slice] - np.nanmin(img_fa_data[:, :, slice])) / (
        #             np.nanmax(img_fa_data[:, :, slice]) - np.nanmin(img_fa_data[:, :, slice]))
        array_t2 = 255 * (img_t2_data[:, :, slice] - np.nanmin(img_t2_data[:, :, slice])) / (
                np.nanmax(img_t2_data[:, :, slice]) - np.nanmin(img_t2_data[:, :, slice]))

        # array_t1 = array_t1.astype(np.uint8)
        # array_fa = array_fa.astype(np.uint8)
        array_t2 = array_t2.astype(np.uint8)

        # empty_t1[:, :, slice] = array_t1
        # empty_fa[:, :, slice] = array_fa
        empty_t2[:, :, slice] = array_t2

    # T1_save_path = data_path + '/new_T1.nii.gz'
    # FA_save_path = data_path + '/new_FA.nii.gz'
    T2_save_path = data_path +'/'+data_num+'_ON-new_T2.nii.gz'

    # t1_data = nib.Nifti1Image(empty_t1, img_t1_affine)
    # fa_data = nib.Nifti1Image(empty_fa, img_fa_affine)
    t2_data = nib.Nifti1Image(empty_t2, img_t2_affine)

    # nib.save(t1_data, T1_save_path)
    # nib.save(fa_data, FA_save_path)
    nib.save(t2_data, T2_save_path)

    print('-' * 30)

if __name__ == '__main__':
    # raw_data_path1 = "/home/AVP数据/初始数据_145x174x145_102/T1/"
    # raw_data_path2 = "/home/AVP数据/初始数据_145x174x145_102/FA/"
    save_train_path = "/home/AVP数据/128x160x128_102/Train_Set_FM/"
    # save_val_path = "/home/AVP数据/128x160x128_102/Val_Set/"
    save_test_path = "/home/AVP数据/128x160x128_102/Test_Set_FM/"
    # raw_clinic_data_path1 = "/home/AVP数据/clinic_data/T1/"
    # raw_clinic_data_path2 = "/home/AVP数据/clinic_data/FA/"
    # raw_clinic_data_path3 = "/home/AVP数据/clinic_data/Peaks/"
    # save_clinic_test_path = "/home/AVP数据/cilinic_testset/"
    dir = os.listdir(save_train_path)  # Get the data number
    x = 0
    for data_num in dir:
        x = x+1
        # data_path = raw_clinic_data_path3+'Peaks_'+data_num+'.nii.gz'
        # cut_data(data_path, save_clinic_test_path, data_num)
        # print(data_num)
        # print(x)
        data_path=save_train_path+data_num
        get_255(data_path, data_num)
        print(data_num)
        print(x)