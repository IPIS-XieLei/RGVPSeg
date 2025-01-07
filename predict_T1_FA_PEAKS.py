import config_2d
from NetModel import Unet_2d, Newnet
import torch
from torch import nn
from torch.utils.data import DataLoader
# from dataset import CN_MyTestDataset
from traindataset_2d import MyTrainDataset
import os
from torchvision.transforms import transforms
import numpy as np
import time
import nibabel as nib
from quantitative_analysis import DICE_count, dice_coef
# from two_model_SAnet import t1safaFuseUNet1, t1sapeaksFuseUNet2
# from SFF_fusion import t1sfffaFuseUNet1, t1sffpeaksFuseUNet2, entire_net
import setproctitle
from networks import MEnet_T1FA

setproctitle.setproctitle("Pre_TTFDP")
unet2d = Unet_2d.UNet2D  # U-Net

os.environ['CUDA_VISIBLE_DEVICES']='0'
# t1safaFuseUNet1 = t1safaFuseUNet1()

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H

batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path
# test_extraction_step = config_2d.TEST_EXTRACTION_STEP
flag_gpu = config_2d.FLAG_GPU
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusionnet = Newnet.DatafusionNet


# 通过模型预测结果
def predict1(img_dir, predict_t1, imgs_num, model1_path):
    global model1, model2, model3, model4, test_dataset, test_dataloader
    # 模型选择

    # model1_path = "outputs3_clinic_sff_fr_t1fa/t1fa_67epoch_32batch.pth"
    # model2_path = "outputs3_clinic_sff_fr_t1peaks/t1peaks_28epoch_32batch.pth"
    # model3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs3_peaks/peaks_14epoch_32batch.pth"

    model1 = MEnet_T1FA.t1safaFuseUNet1(1,1).to(device)
    # model2 = t1sffpeaksFuseUNet2(1, 9).to(device)
    # model3 = unet2d(1, 9).to(device)
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

    test_dataset = MyTrainDataset(ON_test_x_t1_dir, ON_test_x_t2_dir,ON_test_x_fa_dir,ON_test_x_dec_dir, ON_test_x_peaks_dir, ON_test_y_dir,
                                  x_transform=x_transforms, z_transform=z_transforms,
                                  y_transform=y_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_t1_patch_num = 0

    model1.eval()
    # model2.eval()
    # model3.eval()
    # model4.eval()

    with torch.no_grad():
        for x1, x2, x3, x4,x5,y in test_dataloader:
            # inputs1 = x1.to(device)
            inputs2 = x2.to(device)
            inputs3 = x3.to(device)
            # inputs4 = x4.to(device)
            # inputs5 = x5.to(device)

            # input1 = torch.cat([inputs1, inputs2], 1)
            # input2 = torch.cat([inputs3,torch.cat([inputs5,inputs4],1)],1)
            outputs1 = model1(inputs2,inputs3)
            # outputs2 = model2(inputs1,inputs3)
            # outputs3 = model3(inputs3)
            # outputs4 = model4(outputs1, outputs2)
            t1fa_pred = torch.max(outputs1, 1)[1].squeeze()  # troch.max()[1]，只返回最大值的每个索引
            # t1fa_pred_result = t1fa_pred.cpu().numpy().astype(int)
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
    img_name = imgs_num + "_ON-new_T1.nii.gz"

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


def predict2(img_dir, predict_t1, imgs_num, model2_path):
    global model1, model2, model3, model4, test_dataset, test_dataloader
    # 模型选择

    # model1_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_t1/t1_53epoch_32batch.pth"
    # model2_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_fa/fa_64epoch_32batch.pth"
    # model3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_peaks/peaks_100epoch_32batch.pth"

    # model1 = unet2d(1, 2).to(device)
    model2 = MEnet_T1FA.t1safaFuseUNet1(1,9).to(device)
    # model3 = unet2d(9, 2).to(device)
    # model4 = fusionnet(4, 2).to(device)

    # model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()
    # model3 = nn.DataParallel(model3).cuda()
    # model4 = nn.DataParallel(model4).cuda()

    ON_test_x_t1_dir = img_dir + 'x_t1_data/'
    ON_test_x_t2_dir = img_dir + 'x_t2_data/'
    ON_test_x_fa_dir = img_dir + 'x_fa_data/'
    ON_test_x_dec_dir = img_dir + 'x_dec_data/'
    ON_test_x_peaks_dir = img_dir + 'x_peaks_data/'
    ON_test_y_dir = img_dir + 'y_data/'

    # model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
    model2.load_state_dict(torch.load(model2_path, map_location='cpu'))
    # model3.load_state_dict(torch.load(model3_path, map_location='cpu'))
    # model4.load_state_dict(torch.load(model4_path, map_location='cpu'))

    test_dataset = MyTrainDataset(ON_test_x_t1_dir,ON_test_x_t2_dir, ON_test_x_fa_dir,ON_test_x_dec_dir, ON_test_x_peaks_dir, ON_test_y_dir,
                                  x_transform=x_transforms, z_transform=z_transforms,
                                  y_transform=y_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_t1_patch_num = 0

    # model1.eval()
    model2.eval()
    # model3.eval()
    # model4.eval()

    with torch.no_grad():
        for x1, x2, x3,x4,x5, y in test_dataloader:
            inputs1 = x1.to(device)
            # inputs2 = x2.to(device)
            # inputs3 = x3.to(device)
            # inputs4 = x4.to(device)
            inputs5 = x5.to(device)
            # input = torch.cat([inputs1, inputs3], 1)
            # outputs1 = model1(inputs1)
            outputs2 = model2(inputs1, inputs5)
            # outputs3 = model3(inputs3)
            # outputs4 = model4(outputs1, outputs3)
            t1fa_pred = torch.max(outputs2, 1)[1].squeeze()  # troch.max()[1]，只返回最大值的每个索引
            # t1fa_pred_result = t1fa_pred.cpu().numpy().astype(int)
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
    img_name = imgs_num + "_ON-new_T1.nii.gz"

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


def predict3(img_dir, predict_t1, imgs_num, model4_path):
    global model1, model2, model3, model4, test_dataset, test_dataloader
    # 模型选择

    model1_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_t1/t1_53epoch_32batch.pth"
    model2_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_fa/fa_64epoch_32batch.pth"
    model3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_peaks/peaks_100epoch_32batch.pth"
    #
    model1 = unet2d(1, 2).to(device)
    model2 = unet2d(1, 2).to(device)
    model3 = unet2d(9, 2).to(device)
    model4 = fusionnet(6, 2).to(device)

    model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()
    model3 = nn.DataParallel(model3).cuda()
    model4 = nn.DataParallel(model4).cuda()

    ON_test_x_t1_dir = img_dir + 'x_t1_data/'
    ON_test_x_fa_dir = img_dir + 'x_fa_data/'
    ON_test_x_peaks_dir = img_dir + 'x_peaks_data/'
    ON_test_y_dir = img_dir + 'y_data/'

    model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
    model2.load_state_dict(torch.load(model2_path, map_location='cpu'))
    model3.load_state_dict(torch.load(model3_path, map_location='cpu'))
    model4.load_state_dict(torch.load(model4_path, map_location='cpu'))

    test_dataset = MyTrainDataset(ON_test_x_t1_dir, ON_test_x_fa_dir, ON_test_x_peaks_dir, ON_test_y_dir,
                                  x_transform=x_transforms, z_transform=z_transforms,
                                  y_transform=y_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_t1_patch_num = 0

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    with torch.no_grad():
        for x1, x2, x3, y in test_dataloader:
            inputs1 = x1.to(device)
            inputs2 = x2.to(device)
            inputs3 = x3.to(device)
            # input1 = torch.cat([inputs1, inputs2], 1)
            # input2=torch.cat([input1,inputs3],1)
            outputs1 = model1(inputs1)
            outputs2 = model2(inputs2)
            outputs3 = model3(inputs3)
            outputs4 = model4(torch.cat([outputs1, outputs2], 1), outputs3)
            t1fa_pred = torch.max(outputs4, 1)[1].squeeze()  # troch.max()[1]，只返回最大值的每个索引
            # t1fa_pred_result = t1fa_pred.cpu().numpy().astype(int)
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
    pre_seg_t1_final = np.zeros((image_rows, image_cols, image_depth))

    # source.nii
    img_name = imgs_num + '_ON-T1.nii.gz'

    img_name = os.path.join(test_imgs_path, imgs_num, img_name)
    img = nib.load(img_name)
    img_data = img.get_fdata()
    img_affine = img.affine

    # mask.nii
    img_mask_name = imgs_num + '_ON-mask.nii.gz'
    img_mask_name = os.path.join(test_imgs_path, imgs_num, img_mask_name)
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_fdata()
    img_mask_data = np.squeeze(img_mask_data)

    X = img_mask_data.shape

    step = 0

    for iSlice in range(0, X[2]):
        if np.count_nonzero(img_mask_data[:, :, iSlice]) and np.count_nonzero(img_data[:, :, iSlice]):
            pre_name = 'pre_' + str(step) + '.nii.gz'

            pre_t1_name = os.path.join(predict_t1, pre_name)

            pre_seg_t1_temp = nib.load(pre_t1_name)
            pre_seg_t1_temp_data = pre_seg_t1_temp.get_fdata()

            step += 1

            for i in range(0, patch_size_w):
                for j in range(0, patch_size_h):
                    pre_seg_t1_final[i][j][iSlice] = pre_seg_t1_temp_data[i][j]

    pre_seg_t1_final = nib.Nifti1Image(pre_seg_t1_final, img_affine)

    pre_sge_finalname = 'pre_final-label.nii.gz'

    pre_sge_t1_final_savepath = os.path.join(predict_t1, pre_sge_finalname)
    nib.save(pre_seg_t1_final, pre_sge_t1_final_savepath)


## For rough segmentation
if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # x_peaks_transforms = transforms.ToTensor()
    z_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()

    start_time = time.time()

    count = 0
    input_label_base = "/home/AVP数据/128x160x128_102/Test_Set_FM/"

    weight1_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs_MEnet_t2_fa/"
    weights1 = os.listdir(weight1_path)
    weights1.sort(key=lambda x: int(x.split('_')[1].split('e')[0]))

    for weights1 in weights1[0:100]:
        count += 1
        print("现在开始第%d次预测"%count)
        model1_path = weight1_path + '/' + weights1
        test_dir = os.listdir(test_imgs_path) # 01-07
        filename = weights1.split('epoch', -1)[0]

        file_name = "predict_MEnet_t2_fa_fusion_metrics"
        if file_name not in os.listdir(os.curdir):
            os.mkdir(file_name)

        # 预测结果保存
        pre_file_t1 = file_name + '/' + filename  # model1
        # predict1_clinic_2sff_fr_t1fa_fusion_metrics/t1fa_01

        if pre_file_t1 not in os.listdir(os.curdir):
            os.mkdir(pre_file_t1)
        for test_num in test_dir:
            test_name = 'test_' + test_num #test_01
            test_pre_name = 'test_result_' + test_num #test_result_01

            os.mkdir(os.path.join(pre_file_t1, test_pre_name))

            test_input_path = 'ON_mydata/test_data/' + test_name + '/'
            test_result_t1 = pre_file_t1 + '/' + test_pre_name + '/'
            # predict1_clinic_2sff_fr_t1fa_fusion_metrics/t1fa_01/test_result_01/
            ## 1.预测并合成
            predict1(test_input_path, test_result_t1, test_num, model1_path)

        ## 计算dice
        all_dice = []
        all_Iou = []

        # with open(pre_file_t1 + '/' + 'Dice.txt', 'a+') as f:
        #     f.writelines('epoch\n')

        mDice = 0
        Dice = 0

        test_dir = os.listdir(input_label_base) #01-07

        for num in test_dir:
            input1_label_nii = input_label_base + '/' + num + '/' + num + "_ON-label.nii.gz"
            input2_predict_nii = pre_file_t1 + '/' + 'test_result_' + num + '/pre_final-label.nii.gz'

            # DSC
            DICE1 = dice_coef(input1_label_nii, input2_predict_nii)
            Dice = Dice + DICE1
            DICE1 = '%.4f' % (DICE1)
            print('DSC =', DICE1)
            # with open(predice_name + '/' + 'Dice.txt', 'a+') as f:
            #     f.writelines('{0}\t'.format(DICE1))

        mDice = Dice / 20
        mDice = '%.4f' % (mDice)
        print("mDice =", mDice)
        with open('predict_MEnet_t2_fa_fusion_metrics' + '/' + 'Dice.txt', 'a+') as f:
            f.writelines('{0}\t{1}\t\n'.format(filename, mDice))

    # weight2_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs_MEnet_t1_peaks/"
    # weights2 = os.listdir(weight2_path)
    # weights2.sort(key=lambda x: int(x.split('_')[1].split('e')[0]))
    #
    # for weights2 in weights2:
    #     count += 1
    #     print("现在开始第%d次预测" % count)
    #     model2_path = weight2_path + '/' + weights2
    #     test_dir = os.listdir(test_imgs_path)
    #     filename = weights2.split('epoch', -1)[0]
    #
    #     file_name = "predict_MEnet_t1_peaks_fusion_metrics"
    #     if file_name not in os.listdir(os.curdir):
    #         os.mkdir(file_name)
    #
    #     # 预测结果保存
    #     pre_file_t1 = file_name + '/' + filename  # model1
    #
    #     if pre_file_t1 not in os.listdir(os.curdir):
    #         os.mkdir(pre_file_t1)
    #     for test_num in test_dir:
    #         test_name = 'test_' + test_num
    #         test_pre_name = 'test_result_' + test_num
    #
    #         os.mkdir(os.path.join(pre_file_t1, test_pre_name))
    #
    #         test_input_path = 'ON_mydata/test_data/' + test_name + '/'
    #         test_result_t1 = pre_file_t1 + '/' + test_pre_name + '/'
    #
    #         ## 1.预测并合成
    #         predict2(test_input_path, test_result_t1, test_num, model2_path)
    #
    #     ## 计算dice
    #
    #     pre_dir = os.listdir(pre_file_t1 + '/')
    #     input_predict_base = pre_file_t1 + '/'
    #
    #     all_dice = []
    #     all_Iou = []
    #
    #     # with open(pre_file_t1 + '/' + 'Dice.txt', 'a+') as f:
    #     #     f.writelines('epoch\n')
    #
    #     mDice = 0
    #     Dice = 0
    #
    #     input_predict_name = input_predict_base
    #     test_dir = os.listdir(input_label_base)
    #
    #     for num in test_dir:
    #         input1_label_nii = input_label_base + '/' + num + '/' + num + "_ON-label.nii.gz"
    #         input2_predict_nii = input_predict_name + '/' + 'test_result_' + num + '/pre_final-label.nii.gz'
    #
    #         # DSC
    #         DICE1 = dice_coef(input1_label_nii, input2_predict_nii)
    #         Dice = Dice + DICE1
    #         DICE1 = '%.4f' % (DICE1)
    #         print('DSC =', DICE1)
    #         # with open(predice_name + '/' + 'Dice.txt', 'a+') as f:
    #         #     f.writelines('{0}\t'.format(DICE1))
    #
    #     mDice = Dice / 20
    #     mDice = '%.4f' % (mDice)
    #     print("mDice =", mDice)
    #     with open('predict_MEnet_t1_peaks_fusion_metrics' + '/' + 'Dice.txt', 'a+') as f:
    #         f.writelines('{0}\t{1}\t\n'.format(filename, mDice))

    # weight3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_BEFF_t1fapeaks/"
    # weights3 = os.listdir(weight3_path)
    # weights3.sort(key=lambda x: int(x.split('_')[1].split('e')[0]))
    #
    # for weights3 in weights3:
    #     count += 1
    #     print("现在开始第%d次预测" % count)
    #     model3_path = weight3_path + '/' + weights3
    #     test_dir = os.listdir(test_imgs_path)
    #     filename = weights3.split('epoch', -1)[0]
    #
    #     file_name = "predict5_BEFF_t1fapeaks_fusion_metrics"
    #     if file_name not in os.listdir(os.curdir):
    #         os.mkdir(file_name)
    #
    #     # 预测结果保存
    #     pre_file_t1 = file_name + '/' + filename  # model1
    #
    #     if pre_file_t1 not in os.listdir(os.curdir):
    #         os.mkdir(pre_file_t1)
    #     for test_num in test_dir:
    #         test_name = 'test_' + test_num
    #         test_pre_name = 'test_result_' + test_num
    #
    #         os.mkdir(os.path.join(pre_file_t1, test_pre_name))
    #
    #         test_input_path = 'ON_mydata/test_data5/' + test_name + '/'
    #         test_result_t1 = pre_file_t1 + '/' + test_pre_name + '/'
    #
    #         ## 1.预测并合成
    #         predict3(test_input_path, test_result_t1, test_num, model3_path)
    #
    #     ## 计算dice
    #
    #     pre_dir = os.listdir(pre_file_t1 + '/')
    #     input_predict_base = pre_file_t1 + '/'
    #
    #     all_dice = []
    #     all_Iou = []
    #
    #     # with open(pre_file_t1 + '/' + 'Dice.txt', 'a+') as f:
    #     #     f.writelines('epoch\n')
    #
    #     mDice = 0
    #     Dice = 0
    #
    #     input_predict_name = input_predict_base
    #     test_dir = os.listdir(input_label_base)
    #
    #     for num in test_dir:
    #         input1_label_nii = input_label_base + '/' + num + '/' + num + '_ON-label.nii.gz'
    #         input2_predict_nii = input_predict_name + '/' + 'test_result_' + num + '/pre_final-label.nii.gz'
    #
    #         # DSC
    #         DICE1 = dice_coef(input1_label_nii, input2_predict_nii)
    #         Dice = Dice + DICE1
    #         DICE1 = '%.4f' % (DICE1)
    #         print('DSC =', DICE1)
    #         # with open(predice_name + '/' + 'Dice.txt', 'a+') as f:
    #         #     f.writelines('{0}\t'.format(DICE1))
    #
    #     mDice = Dice / 20
    #     mDice = '%.4f' % (mDice)
    #     print("mDice =", mDice)
    #     with open('predict5_BEFF_t1fapeaks_fusion_metrics' + '/' + 'Dice.txt', 'a+') as f:
    #         f.writelines('{0}\t{1}\t\n'.format(filename, mDice))

    end_time = time.time()
    print("2D train time is {:.3f} s".format((end_time - start_time)))
