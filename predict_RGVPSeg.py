import shutil

import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d, Newnet,SAnet
import torch
from torch import nn
from torch.utils.data import DataLoader
from traindataset_2d import MyTrainDataset
import os
from torchvision.transforms import transforms
import numpy as np
import time
import nibabel as nib
from skimage import measure
from SFF_fusion import t1sfffaFuseUNet1, t1sffpeaksFuseUNet2
from networks import MEnet_T1FA

unet2d = Unet_2d.UNet2D  # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D  # U-Net++
multiresunet2d = MultiResUnet_2d.MultiResUnet2D  # MultiRes U-Net

fusionnet = Newnet.DatafusionNet

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
flag_gpu = config_2d.FLAG_GPU

batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path
test_extraction_step = config_2d.TEST_EXTRACTION_STEP
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add(inputs1, inputs2):
    intput = torch.cat([inputs1, inputs2], 1)
    return intput
# 通过模型预测结果
def predict(img_dir, predict_t1, imgs_num):
    global model1, model2, model3, model4, test_dataset, test_dataloader

    model1_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs_MEnet_t1_fa/t1fa_95epoch_32batch.pth"
    # model2_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_clinic_fa/fa_89epoch_32batch.pth"
    # model3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_clinic_FECC_t1fa/t1fa_66epoch_32batch.pth"
    # model4_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_clinic_BEFF_t1fa/t1fa_71epoch_32batch.pth"

    # 模型选择
    model1 = MEnet_T1FA.t1safaFuseUNet1(2).to(device)
    # model2 = unet2d(1, 2).to(device)
    # model3 = unet2d(2, 2).to(device)
    # model4 = fusionnet(4, 2).to(device)

    model1 = nn.DataParallel(model1).cuda()
    # model2 = nn.DataParallel(model2).cuda()
    # model3 = nn.DataParallel(model3).cuda()
    # model4 = nn.DataParallel(model4).cuda()

    ON_test_x_t1_dir = img_dir + 'x_t1_data/'
    ON_test_x_t2_dir = img_dir + 'x_t2_data/'
    ON_test_x_fa_dir = img_dir + 'x_fa_data/'
    ON_test_x_dec_dir = img_dir + 'x_dec_data/'
    ON_test_x_peaks_dir = img_dir + 'x_peaks_data/'
    ON_test_y_dir = img_dir + 'y_data/'

    model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
    # model2.load_state_dict(torch.load(model2_path, map_location='cpu'))
    # model3.load_state_dict(torch.load(model3_path, map_location='cpu'))
    # model4.load_state_dict(torch.load(model4_path, map_location='cpu'))

    test_dataset = MyTrainDataset(ON_test_x_t1_dir,ON_test_x_t2_dir, ON_test_x_fa_dir, ON_test_x_dec_dir, ON_test_x_peaks_dir, ON_test_y_dir,
                                  x_transform=x_transforms, z_transform=z_transforms, y_transform=y_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_t1_patch_num = 0

    model1.eval()
    # model2.eval()
    # model3.eval()
    # model4.eval()

    with torch.no_grad():
        for x1, x2, x3, x4, x5, y in test_dataloader:
            inputs1 = x1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            # inputs2 = x2.to(device)
            inputs3 = x3.to(device)
            # inputs4 = x4.to(device)
            # inputs5 = x5.to(device)
            # input_t1fa = torch.cat((inputs1, inputs2), dim=1)
            # input_t1peaks = torch.cat((inputs1, inputs3), dim=1)
            # input_t1fapeaks = torch.cat((input_t1fa, inputs3), dim=1)
            outputs1 = model1(inputs1,inputs3)
            # outputs2 = model2(inputs2)
            # outputs3 = model3(input_t1fa)
            # pre_final = model4(outputs1, outputs2)
            t1fa_pred = torch.max(outputs1, 1)[1].squeeze()  # troch.max()[1]，只返回最大值的每个索引
            t1fa_pred_result = t1fa_pred.cpu().numpy().astype(float)

            if all_t1_patch_num == 0:
                test_t1_data_path = ON_test_x_t1_dir + 'x_t1-data_0.nii.gz'
                test_t1_img = nib.load(test_t1_data_path)
                test_t1_img_affine = test_t1_img.affine
            t1_pred_patches_nii = nib.Nifti1Image(t1fa_pred_result, test_t1_img_affine)
            t1_pred_nii_path = predict_t1 + 'pre_' + str(all_t1_patch_num) + '.nii.gz'
            nib.save(t1_pred_patches_nii, t1_pred_nii_path)
            all_t1_patch_num += 1

    #### -------------------------------------------------
    #### -------------------------------------------------
    #### -------------------------------------------------
    ## Combination
    pre_seg_t1fa_final = np.zeros((image_rows, image_cols, image_depth))

    # source.nii
    img_name = imgs_num +  "_ON-new_T1.nii.gz"

    img_name = os.path.join(test_imgs_path, imgs_num, img_name)
    img = nib.load(img_name)
    img_data = img.get_fdata()
    img_affine = img.affine

    # mask.nii
    img_mask_name = imgs_num + "_ON-mask.nii.gz"  # 根据测试集取的mask来调整
    img_mask_name = os.path.join(test_imgs_path, imgs_num, img_mask_name)
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_fdata()
    img_mask_data = np.squeeze(img_mask_data)

    X = img_mask_data.shape

    step = 0

    for iSlice in range(0, X[2]):
        if np.count_nonzero(img_mask_data[:, :, iSlice]) and np.count_nonzero(img_data[:, :, iSlice]):
            pre_name = 'pre_' + str(step) + '.nii.gz'

            pre_t1fa_name = os.path.join(predict_t1, pre_name)

            pre_seg_t1fa_temp = nib.load(pre_t1fa_name)
            pre_seg_t1fa_temp_data = pre_seg_t1fa_temp.get_fdata()

            step += 1

            for i in range(0, patch_size_w):
                for j in range(0, patch_size_h):
                    pre_seg_t1fa_final[i][j][iSlice] = pre_seg_t1fa_temp_data[i][j]

    pre_seg_t1fa_final = nib.Nifti1Image(pre_seg_t1fa_final, img_affine)

    pre_sge_finalname = 'pre_final-label.nii.gz'

    pre_sge_t1fa_final_savepath = os.path.join(predict_t1, pre_sge_finalname)
    nib.save(pre_seg_t1fa_final, pre_sge_t1fa_final_savepath)


if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    z_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()

    # 预测结果保存
    pre_file_t1fa = 'predict_MEnet_t1fa_fusion_mask'  # model3
    # shutil.rmtree('predict4_FECC_t1fa_fusion_mask')

    if pre_file_t1fa not in os.listdir(os.curdir):
        os.mkdir(pre_file_t1fa)

    start_time = time.time()

    test_dir = os.listdir(test_imgs_path)

    for test_num in test_dir:
        test_name = 'test_' + test_num
        test_pre_name = 'test_result_' + test_num

        os.mkdir(os.path.join(pre_file_t1fa, test_pre_name))

        test_input_path = 'ON_mydata/test_data/' + test_name + '/'
        test_result_t1fa = pre_file_t1fa + '/' + test_pre_name + '/'

        ## 1.预测并合成
        predict(test_input_path, test_result_t1fa, test_num)

    end_time = time.time()
    print("2D train time is {:.3f} s".format((end_time - start_time)))
