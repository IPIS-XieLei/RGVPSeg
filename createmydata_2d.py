import numpy as np
import config_2d
import torch
import time, os
import nibabel as nib

batch_size = config_2d.BATCH_SIZE
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
n_classes = config_2d.NUM_CLASSES

train_imgs_path = config_2d.train_imgs_path
# val_imgs_path = config_2d.val_imgs_path
test_imgs_path = config_2d.test_imgs_path
# test_clinic_imgs_path = config_2d.test_clinic_imgs_path

train_save_path = 'ON_mydata/train_data'
# val_save_path = 'ON_mydata/val_data'
test_inputdata = 'ON_mydata/test_data'
# test_inputdata_clinic = 'ON_mydata/test_clinic_data'


##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##### create train data 训练集数据 ####
def create_train_data(imgs_path, save_path):
    images_dir = os.listdir(imgs_path)
    x_t1_dir = os.path.join(save_path, 'x_t1_data')  # CTL(T1)、CTL(FA)
    x_t2_dir = os.path.join(save_path, 'x_t2_data')
    x_dec_dir = os.path.join(save_path, 'x_dec_data')
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir = os.path.join(save_path, 'y_data')

    j = 0
    print('-' * 30)
    print('Creating train 2d_patches...')
    print('-' * 30)

    all_patch_num = 0

    num = 0
    # for each volume do:
    for img_dir_name in images_dir:
        num = num + 1
        # T1
        img_t1_name = img_dir_name + '_ON-new_T1.nii.gz'  # T1
        img_t1_name = os.path.join(imgs_path, img_dir_name, img_t1_name)

        img_t2_name = img_dir_name + '_ON-new_T2.nii.gz'
        img_t2_name = os.path.join(imgs_path, img_dir_name, img_t2_name)

        img_dec_name = img_dir_name + '_ON-DEC.nii.gz'
        img_dec_name = os.path.join(imgs_path, img_dir_name, img_dec_name)
        # FA
        img_t1_fa_name = img_dir_name + '_ON-new_FA.nii.gz'  # FA
        img_t1_fa_name = os.path.join(imgs_path, img_dir_name, img_t1_fa_name)
        # PEAKS
        img_t1_peaks_name = img_dir_name + '_ON-Peaks.nii.gz'  # PEAKS
        img_t1_peaks_name = os.path.join(imgs_path, img_dir_name, img_t1_peaks_name)
        # label
        img_label_name = img_dir_name + '_ON-label.nii.gz'
        img_label_name = os.path.join(imgs_path, img_dir_name, img_label_name)
        # mask
        img_mask_name = img_dir_name + '_ON-mask.nii.gz'
        img_mask_name = os.path.join(imgs_path, img_dir_name, img_mask_name)

        # load T1, FA, PEAKS, label and mask
        img_t1 = nib.load(img_t1_name)
        img_t1_data = img_t1.get_fdata()
        img_t1_affine = img_t1.affine
        img_t1_data = np.squeeze(img_t1_data)
        # T2.nii
        img_t2 = nib.load(img_t2_name)
        img_t2_affine = img_t2.affine
        img_t2_data = img_t2.get_fdata()
        img_t2_data = np.squeeze(img_t2_data)
        # DEC.nii
        img_dec = nib.load(img_dec_name)
        img_dec_affine = img_dec.affine
        img_dec_data = img_dec.get_fdata()
        img_dec_data = np.squeeze(img_dec_data)

        img_fa = nib.load(img_t1_fa_name)
        img_fa_data = img_fa.get_fdata()
        img_fa_affine = img_fa.affine
        img_fa_data = np.squeeze(img_fa_data)

        img_peaks = nib.load(img_t1_peaks_name)
        img_peaks_data = img_peaks.get_fdata()
        img_peaks_affine = img_peaks.affine
        img_peaks_data = np.squeeze(img_peaks_data)

        img_label = nib.load(img_label_name)
        img_label_data = img_label.get_fdata()
        img_label_affine = img_label.affine
        img_label_data = np.squeeze(img_label_data)

        img_mask = nib.load(img_mask_name)
        img_mask_data = img_mask.get_fdata()
        img_mask_data = np.squeeze(img_mask_data)

        X = img_label_data.shape

        # for each slice do
        for slice in range(X[2]):
            print('Processing: volume {0} / {1} volume images, slice {2} / {3} slices'.format(j + 1,
                                                                                              len(images_dir),
                                                                                              slice + 1,
                                                                                              img_label_data.shape[2]))

            # 2D Axial

            # if np.count_nonzero(img_mask_data[:, :, slice]) and np.count_nonzero(img_t1_data[:, :, slice]):
            # if np.count_nonzero(img_label_data[:, :, slice]) and np.count_nonzero(img_t1_data[:, :, slice]):
            if np.count_nonzero(img_label_data[:, :, slice]) >= 0 and np.count_nonzero(img_mask_data[:, :, slice]) >= 0 and np.count_nonzero(img_t1_data[:, :, slice]) >= 0:
                t1_patches = img_t1_data[:, :, slice]
                t2_patches = img_t2_data[:, :, slice]
                dec_patches = img_dec_data[:, :, slice, :]
                fa_patches = img_fa_data[:, :, slice]
                peaks_patches = img_peaks_data[:, :, slice, :]
                label_patches = img_label_data[:, :, slice]
                # x_t1 data
                t1_patches_nii = t1_patches
                t1_flip_patches_nii = np.flip(t1_patches_nii, 0)
                t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
                t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
                nib.save(t1_patches_nii, t1_nonum_nii_path)
                # x_t2 data
                t2_patches_nii = t2_patches
                t2_flip_patches_nii = np.flip(t2_patches_nii, 0)
                t2_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(all_patch_num) + '.nii.gz'
                t2_patches_nii = nib.Nifti1Image(t2_patches_nii, img_t2_affine)
                nib.save(t2_patches_nii, t2_nonum_nii_path)
                # x_dec data
                dec_patches_nii = dec_patches
                dec_flip_patches_nii = np.flip(dec_patches_nii, 0)
                dec_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(all_patch_num) + '.nii.gz'
                dec_patches_nii = nib.Nifti1Image(dec_patches_nii, img_dec_affine)
                nib.save(dec_patches_nii, dec_nonum_nii_path)
                # x_fa data
                fa_patches_nii = fa_patches
                fa_flip_patches_nii = np.flip(fa_patches_nii, 0)
                fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
                fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
                nib.save(fa_patches_nii, fa_nonum_nii_path)
                # x_peaks data
                peaks_patches_nii = peaks_patches
                peaks_flip_patches_nii = np.flip(peaks_patches_nii, 0)
                peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
                peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
                nib.save(peaks_patches_nii, peaks_nonum_nii_path)
                # y_data
                label_patches_nii = label_patches
                label_flip_patches_nii = np.flip(label_patches_nii, 0)
                label_nonum_nii_path = y_dir + '/y-data_' + str(all_patch_num) + '.nii.gz'
                label_patches_nii = nib.Nifti1Image(label_patches_nii, img_label_affine)
                nib.save(label_patches_nii, label_nonum_nii_path)
                all_patch_num += 1

                ### --------------
                ###数据增强
                ### t1 x flip data  （翻转180）
                t1_flip_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
                t1_flip_patches_nii = nib.Nifti1Image(t1_flip_patches_nii, img_t1_affine)
                nib.save(t1_flip_patches_nii, t1_flip_nonum_nii_path)
                ### t2 x flip data  （翻转180）
                t2_flip_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(all_patch_num) + '.nii.gz'
                t2_flip_patches_nii = nib.Nifti1Image(t2_flip_patches_nii, img_t2_affine)
                nib.save(t2_flip_patches_nii, t2_flip_nonum_nii_path)
                ### fa x flip data  （翻转180）
                fa_flip_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
                fa_flip_patches_nii = nib.Nifti1Image(fa_flip_patches_nii, img_fa_affine)
                nib.save(fa_flip_patches_nii, fa_flip_nonum_nii_path)
                ### dec x flip data  （翻转180）
                dec_flip_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(all_patch_num) + '.nii.gz'
                dec_flip_patches_nii = nib.Nifti1Image(dec_flip_patches_nii, img_dec_affine)
                nib.save(dec_flip_patches_nii, dec_flip_nonum_nii_path)
                ### peaks x flip data  （翻转180）
                peaks_flip_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
                peaks_flip_patches_nii = nib.Nifti1Image(peaks_flip_patches_nii, img_peaks_affine)
                nib.save(peaks_flip_patches_nii, peaks_flip_nonum_nii_path)
                ### y flip data
                label_flip_nonum_nii_path = y_dir + '/y-data_' + str(all_patch_num) + '.nii.gz'
                label_flip_patches_nii = nib.Nifti1Image(label_flip_patches_nii, img_label_affine)
                nib.save(label_flip_patches_nii, label_flip_nonum_nii_path)
                all_patch_num += 1
                ### --------------

        j += 1
        print('Input num:   {0}'.format(img_dir_name))
        print('All Patch num:  {0}'.format(all_patch_num))
        print('Patch size:  [{0}*{1}]'.format(patch_size_w, patch_size_h))
    print('-' * 30)
    print('All Patches: {0}'.format(all_patch_num))
    print('-' * 30)


##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##### create val data 创建验证集数据 ####
def create_val_data(imgs_path, save_path):
    images_dir = os.listdir(imgs_path)
    x_t1_dir = os.path.join(save_path, 'x_t1_data')  # CTL(T1)、CTL(FA)
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir = os.path.join(save_path, 'y_data')

    j = 0
    print('-' * 30)
    print('Creating train 2d_patches...')
    print('-' * 30)

    all_patch_num = 0

    num = 0
    # for each volume do:
    for img_dir_name in images_dir:
        num = num + 1
        # T1
        img_t1_name = img_dir_name + '_ON-new_T1.nii.gz'  # T1
        img_t1_name = os.path.join(imgs_path, img_dir_name, img_t1_name)
        # FA
        img_t1_fa_name = img_dir_name + '_ON-new_FA.nii.gz'  # FA
        img_t1_fa_name = os.path.join(imgs_path, img_dir_name, img_t1_fa_name)
        # PEAKS
        img_t1_peaks_name = img_dir_name + '_ON-Peaks.nii.gz'  # PEAKS
        img_t1_peaks_name = os.path.join(imgs_path, img_dir_name, img_t1_peaks_name)
        # label
        img_label_name = img_dir_name + '_ON-label.nii.gz'
        img_label_name = os.path.join(imgs_path, img_dir_name, img_label_name)
        # mask
        img_mask_name = img_dir_name + '_ON-mask.nii.gz'
        img_mask_name = os.path.join(imgs_path, img_dir_name, img_mask_name)

        # load T1, FA, PEAKS, label and mask
        img_t1 = nib.load(img_t1_name)
        img_t1_data = img_t1.get_fdata()
        img_t1_affine = img_t1.affine
        img_t1_data = np.squeeze(img_t1_data)

        img_fa = nib.load(img_t1_fa_name)
        img_fa_data = img_fa.get_fdata()
        img_fa_affine = img_fa.affine
        img_fa_data = np.squeeze(img_fa_data)

        img_peaks = nib.load(img_t1_peaks_name)
        img_peaks_data = img_peaks.get_fdata()
        img_peaks_affine = img_peaks.affine
        img_peaks_data = np.squeeze(img_peaks_data)

        img_label = nib.load(img_label_name)
        img_label_data = img_label.get_fdata()
        img_label_affine = img_label.affine
        img_label_data = np.squeeze(img_label_data)

        img_mask = nib.load(img_mask_name)
        img_mask_data = img_mask.get_fdata()
        img_mask_data = np.squeeze(img_mask_data)

        X = img_label_data.shape

        # for each slice do
        for slice in range(X[2]):
            print('Processing: volume {0} / {1} volume images, slice {2} / {3} slices'.format(j + 1,
                                                                                              len(images_dir),
                                                                                              slice + 1,
                                                                                              img_label_data.shape[2]))

            # 2D Axial
            # if np.count_nonzero(img_mask_data[:, :, slice]) and np.count_nonzero(img_t1_data[:, :, slice]):
            # if np.count_nonzero(img_label_data[:, :, slice]) and np.count_nonzero(img_t1_data[:, :, slice]):
            if np.count_nonzero(img_label_data[:, :, slice]) >= 0 and np.count_nonzero(
                    img_mask_data[:, :, slice]) >= 0 and np.count_nonzero(img_t1_data[:, :, slice]) >= 0:
                t1_patches = img_t1_data[:, :, slice]
                fa_patches = img_fa_data[:, :, slice]
                peaks_patches = img_peaks_data[:, :, slice, :]
                label_patches = img_label_data[:, :, slice]
                # x_t1 data
                t1_patches_nii = t1_patches
                t1_flip_patches_nii = np.flip(t1_patches_nii, 0)
                t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
                t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
                nib.save(t1_patches_nii, t1_nonum_nii_path)
                # x_fa data
                fa_patches_nii = fa_patches
                fa_flip_patches_nii = np.flip(fa_patches_nii, 0)
                fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
                fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
                nib.save(fa_patches_nii, fa_nonum_nii_path)
                # x_peaks data
                peaks_patches_nii = peaks_patches
                peaks_flip_patches_nii = np.flip(peaks_patches_nii, 0)
                peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
                peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
                nib.save(peaks_patches_nii, peaks_nonum_nii_path)
                # y_data
                label_patches_nii = label_patches
                label_flip_patches_nii = np.flip(label_patches_nii, 0)
                label_nonum_nii_path = y_dir + '/y-data_' + str(all_patch_num) + '.nii.gz'
                label_patches_nii = nib.Nifti1Image(label_patches_nii, img_label_affine)
                nib.save(label_patches_nii, label_nonum_nii_path)
                all_patch_num += 1
        j += 1
        print('Input num:   {0}'.format(img_dir_name))
        print('All Patch num:  {0}'.format(all_patch_num))
        print('Patch size:  [{0}*{1}]'.format(patch_size_w, patch_size_h))
    print('-' * 30)
    print('All Patches: {0}'.format(all_patch_num))
    print('-' * 30)


##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##---------------------------------------------------------------------------------
##### create test data 创建测试集数据 ####
def create_test_data(img_dir_name, save_path):
    x_t1_dir = os.path.join(save_path, 'x_t1_data')  # CTL(T1)、CTL(FA)
    x_t2_dir = os.path.join(save_path, 'x_t2_data')
    x_dec_dir = os.path.join(save_path, 'x_dec_data')
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir = os.path.join(save_path, 'y_data')

    j = 0
    print('-' * 30)
    print('Creating test 2d_patches...')
    print('-' * 30)

    # for each volume do:
    img_t1_name = img_dir_name + '_ON-new_T1.nii.gz'
    img_t1_name = os.path.join(test_imgs_path, img_dir_name, img_t1_name)

    img_t2_name = img_dir_name + '_ON-new_T2.nii.gz'
    img_t2_name = os.path.join(test_imgs_path, img_dir_name, img_t2_name)

    img_dec_name = img_dir_name + '_ON-DEC.nii.gz'
    img_dec_name = os.path.join(test_imgs_path, img_dir_name, img_dec_name)

    img_fa_name = img_dir_name + '_ON-new_FA.nii.gz'
    img_fa_name = os.path.join(test_imgs_path, img_dir_name, img_fa_name)

    img_peaks_name = img_dir_name + '_ON-Peaks.nii.gz'
    img_peaks_name = os.path.join(test_imgs_path, img_dir_name, img_peaks_name)

    img_label_name = img_dir_name + '_ON-label.nii.gz'
    img_label_name = os.path.join(test_imgs_path, img_dir_name, img_label_name)

    img_mask_name = img_dir_name + '_ON-mask.nii.gz'  # the whole brain mask
    img_mask_name = os.path.join(test_imgs_path, img_dir_name, img_mask_name)

    # T1.nii
    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.affine
    img_t1_data = img_t1.get_fdata()
    img_t1_data = np.squeeze(img_t1_data)
    # T2.nii
    img_t2 = nib.load(img_t2_name)
    img_t2_affine = img_t2.affine
    img_t2_data = img_t2.get_fdata()
    img_t2_data = np.squeeze(img_t2_data)
    # DEC.nii
    img_dec = nib.load(img_dec_name)
    img_dec_affine = img_dec.affine
    img_dec_data = img_dec.get_fdata()
    img_dec_data = np.squeeze(img_dec_data)
    # FA.nii
    img_fa = nib.load(img_fa_name)
    img_fa_data = img_fa.get_fdata()
    img_fa_affine = img_fa.affine
    img_fa_data = np.squeeze(img_fa_data)
    # PEAKS.nii
    img_peaks = nib.load(img_peaks_name)
    img_peaks_data = img_peaks.get_fdata()
    img_peaks_affine = img_peaks.affine
    img_peaks_data = np.squeeze(img_peaks_data)
    # label.nii
    img_label = nib.load(img_label_name)
    img_label_data = img_label.get_fdata()
    img_label_affine = img_label.affine
    img_label_data = np.squeeze(img_label_data)
    # mask.nii
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_fdata()
    img_mask_data = np.squeeze(img_mask_data)

    X = img_label_data.shape

    all_patch_num = 0

    for slice in range(X[2]):
        # 2D Axial
        if np.count_nonzero(img_mask_data[:, :, slice]) >= 0 and np.count_nonzero(img_label_data[:, :, slice]) >= 0 and np.count_nonzero(img_t1_data[:, :, slice]) >= 0:
            t1_patches = img_t1_data[:, :, slice]
            t2_patches = img_t2_data[:, :, slice]
            dec_patches = img_dec_data[:, :, slice,:]
            fa_patches = img_fa_data[:, :, slice]
            peaks_patches = img_peaks_data[:, :, slice, :]
            label_patches = img_label_data[:, :, slice]
            # x_t1 data
            t1_patches_nii = t1_patches
            t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
            t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
            nib.save(t1_patches_nii, t1_nonum_nii_path)
            # x_t2 data
            t2_patches_nii = t2_patches
            t2_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(all_patch_num) + '.nii.gz'
            t2_patches_nii = nib.Nifti1Image(t2_patches_nii, img_t2_affine)
            nib.save(t2_patches_nii, t2_nonum_nii_path)
            # x_dec data
            dec_patches_nii = dec_patches
            dec_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(all_patch_num) + '.nii.gz'
            dec_patches_nii = nib.Nifti1Image(dec_patches_nii, img_dec_affine)
            nib.save(dec_patches_nii, dec_nonum_nii_path)
            # x_fa data
            fa_patches_nii = fa_patches
            fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
            fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
            nib.save(fa_patches_nii, fa_nonum_nii_path)
            # x_peaks data
            peaks_patches_nii = peaks_patches
            peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
            peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
            nib.save(peaks_patches_nii, peaks_nonum_nii_path)
            # y_data
            label_patches_nii = label_patches
            label_nonum_nii_path = y_dir + '/y-data_' + str(all_patch_num) + '.nii.gz'
            label_patches_nii = nib.Nifti1Image(label_patches_nii, img_label_affine)
            nib.save(label_patches_nii, label_nonum_nii_path)
            all_patch_num += 1
            ### --------------
            ### --------------
    j += 1
    print('-' * 30)

def create_test_clinic_data(img_dir_name, save_path):
    x_t1_dir = os.path.join(save_path, 'x_t1_data')  # CTL(T1)、CTL(FA)
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir = os.path.join(save_path, 'y_data')

    j = 0
    print('-' * 30)
    print('Creating test 2d_patches...')
    print('-' * 30)

    # for each volume do:
    img_t1_name = img_dir_name + '/new_T1.nii.gz'
    img_t1_name = os.path.join(test_clinic_imgs_path, img_t1_name)

    img_fa_name = img_dir_name + '/new_FA.nii.gz'
    img_fa_name = os.path.join(test_clinic_imgs_path, img_fa_name)

    img_peaks_name = img_dir_name + '/Peaks_'+img_dir_name+'.nii.gz'
    img_peaks_name = os.path.join(test_clinic_imgs_path, img_peaks_name)

    img_label_name = img_dir_name + '/Label_'+img_dir_name+'.nii.gz'
    img_label_name = os.path.join(test_clinic_imgs_path, img_label_name)

    img_mask_name = img_dir_name + '/mask_'+img_dir_name+'.nii.gz'  # the whole brain mask
    img_mask_name = os.path.join(test_clinic_imgs_path, img_mask_name)

    # T1.nii
    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.affine
    img_t1_data = img_t1.get_fdata()
    img_t1_data = np.squeeze(img_t1_data)
    # FA.nii
    img_fa = nib.load(img_fa_name)
    img_fa_data = img_fa.get_fdata()
    img_fa_affine = img_fa.affine
    img_fa_data = np.squeeze(img_fa_data)
    # PEAKS.nii
    img_peaks = nib.load(img_peaks_name)
    img_peaks_data = img_peaks.get_fdata()
    img_peaks_affine = img_peaks.affine
    img_peaks_data = np.squeeze(img_peaks_data)
    # label.nii
    img_label = nib.load(img_label_name)
    img_label_data = img_label.get_fdata()
    img_label_affine = img_label.affine
    img_label_data = np.squeeze(img_label_data)
    # mask.nii
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_fdata()
    img_mask_data = np.squeeze(img_mask_data)

    X = img_label_data.shape

    all_patch_num = 0

    for slice in range(X[2]):
        # 2D Axial
        if np.count_nonzero(img_mask_data[:, :, slice]) >= 0 and np.count_nonzero(img_label_data[:, :, slice]) >= 0 and np.count_nonzero(img_t1_data[:, :, slice]) >= 0:
            t1_patches = img_t1_data[:, :, slice]
            fa_patches = img_fa_data[:, :, slice]
            peaks_patches = img_peaks_data[:, :, slice, :]
            label_patches = img_label_data[:, :, slice]
            # x_t1 data
            t1_patches_nii = t1_patches
            t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
            t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
            nib.save(t1_patches_nii, t1_nonum_nii_path)
            # x_fa data
            fa_patches_nii = fa_patches
            fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
            fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
            nib.save(fa_patches_nii, fa_nonum_nii_path)
            # x_peaks data
            peaks_patches_nii = peaks_patches
            peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
            peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
            nib.save(peaks_patches_nii, peaks_nonum_nii_path)
            # y_data
            label_patches_nii = label_patches
            label_nonum_nii_path = y_dir + '/y-data_' + str(all_patch_num) + '.nii.gz'
            label_patches_nii = nib.Nifti1Image(label_patches_nii, img_label_affine)
            nib.save(label_patches_nii, label_nonum_nii_path)
            all_patch_num += 1
            ### --------------
            ### --------------
    j += 1
    print('-' * 30)



if __name__ == '__main__':
    if 'ON_mydata' not in os.listdir(os.curdir):
        os.mkdir('ON_mydata')
    # #### 1. create my train data: ###
    trainfile_name = 'train_data'
    if trainfile_name not in os.listdir('ON_mydata'):
        os.mkdir(os.path.join('ON_mydata', trainfile_name))
    if 'x_t1_data' not in os.listdir('ON_mydata/'+ trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_t1_data'))
    if 'x_t2_data' not in os.listdir('ON_mydata/'+ trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_t2_data'))
    if 'x_dec_data' not in os.listdir('ON_mydata/'+ trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_dec_data'))
    if 'x_fa_data' not in os.listdir('ON_mydata/' + trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_fa_data'))
    if 'x_peaks_data' not in os.listdir('ON_mydata/' + trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_peaks_data'))
    if 'y_data' not in os.listdir('ON_mydata/' + trainfile_name):
        os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'y_data'))
    create_train_data(train_imgs_path, train_save_path)


    # ### 2. create my val data: ###
    # valfile_name = 'val_data'
    # if valfile_name not in os.listdir('ON_mydata'):
    #     os.mkdir(os.path.join('ON_mydata', valfile_name))
    # if 'x_t1_data' not in os.listdir('ON_mydata/' + valfile_name):
    #     os.mkdir(os.path.join('ON_mydata/' + valfile_name, 'x_t1_data'))
    # if 'x_fa_data' not in os.listdir('ON_mydata/' + valfile_name):
    #     os.mkdir(os.path.join('ON_mydata/' + valfile_name, 'x_fa_data'))
    # if 'x_peaks_data' not in os.listdir('ON_mydata/' + valfile_name):
    #     os.mkdir(os.path.join('ON_mydata/' + valfile_name, 'x_peaks_data'))
    # if 'y_data' not in os.listdir('ON_mydata/' + valfile_name):
    #     os.mkdir(os.path.join('ON_mydata/' + valfile_name, 'y_data'))
    # create_val_data(val_imgs_path, val_save_path)

    ## 3. creatr my test data(one by one): ###
    # testfile_name = 'test_data'  # test  ...
    # if testfile_name not in os.listdir('ON_mydata'):
    #     os.mkdir(os.path.join('ON_mydata', testfile_name))
    # test_dir = os.listdir(test_imgs_path)
    # for test_num in test_dir:
    #     test_name = 'test_' + test_num
    #     test_inputdata_save_path = os.path.join(test_inputdata, test_name)
    #     os.mkdir(test_inputdata_save_path)
    #     if 'x_t1_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_t1_data'))
    #     if 'x_t2_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_t2_data'))
    #     if 'x_dec_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_dec_data'))
    #     if 'x_fa_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_fa_data'))
    #     if 'x_peaks_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_peaks_data'))
    #     if 'y_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data'))
    #     create_test_data(test_num, test_inputdata_save_path)

    ## 4. creatr my clinic test data(one by one): ###
    # testfile_name = 'test_clinic_data'  # test  ...
    # if testfile_name not in os.listdir('ON_mydata'):
    #     os.mkdir(os.path.join('ON_mydata', testfile_name))
    # test_dir = os.listdir(test_clinic_imgs_path)
    # for test_num in test_dir:
    #     test_name = 'test_' + test_num
    #     test_inputdata_save_path = os.path.join(test_inputdata_clinic, test_name)
    #     # os.mkdir(test_inputdata_save_path)
    #     if 'x_t1_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_t1_data'))
    #     if 'x_fa_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_fa_data'))
    #     if 'x_peaks_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_peaks_data'))
    #     if 'y_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data'))
    #     create_test_clinic_data(test_num, test_inputdata_save_path)
