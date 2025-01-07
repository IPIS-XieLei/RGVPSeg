import nibabel as nib
import numpy as np
import os
import scipy.signal as signal
import scipy as sp
import scipy.ndimage
import SimpleITK as sitk
import config_2d
# from skimage import measure
import math
from multiprocessing.dummy import Pool as ThreadPool
import cmath

test_imgs_path = config_2d.test_imgs_path

batch_size = config_2d.BATCH_SIZE



def DICE_count(input1, input2):
    '''
    :param input1:   "107018_ON-label.nii"
    :param intput2:  "107018_ON-predict.nii"
    :return:
    '''
    smooth = 1.
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()

    if label_load.shape != predict_load.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    VsegandVgt = 0
    Vseg = 0
    Vgt = 0

    #for k in range(z):#25-55
    for k in range(0,128):
        im1 = np.squeeze(img1[:, :, k] > 0.5)
        im2 = np.squeeze(img2[:, :, k] > 0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        segandgt = (np.logical_and(im1, im2)).sum() #两个1才为1（取交集）
        Sgt = im1.sum() #每层真实值部分
        Sseg = im2.sum()#每层预测值部分

        VsegandVgt = VsegandVgt + segandgt
        Vgt = Vgt + Sgt   #真实值为1体素个数
        Vseg = Vseg + Sseg #预测值为1体素个数

    #DICE = 2. * (VsegandVgt) /(Vseg + Vgt)
    DICE = (2. * VsegandVgt + smooth) / (Vseg + Vgt + smooth)
    # print("VsegandVgt={}".format(VsegandVgt))
    # print("Vgt={}".format(Vgt))
    # print("Vseg={}".format(Vseg))
    # print("DICE=", DICE)
    # print("done!")
    # print('-' * 30)
    return DICE


def VOE_count(input1, input2):
    '''
    :param input1:   "107018_ON-label.nii"
    :param intput2:  "107018_ON-predict.nii"
    :return:
    '''
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()

    if label_load.shape != predict_load.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    Vseg = 0
    Vgt = 0

    # for k in range(z):#25-55
    for k in range(0, 128):
        im1 = np.squeeze(img1[:, :, k] > 0.5)
        im2 = np.squeeze(img2[:, :, k] > 0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        Sgt = im1.sum()  # 每层真实值部分
        Sseg = im2.sum()  # 每层预测值部分

        Vgt = Vgt + Sgt  # 真实值为1体素个数
        Vseg = Vseg + Sseg  # 预测值为1体素个数

    VOE = 2. * (Vseg - Vgt) / (Vseg + Vgt)
    # print("Vgt={}".format(Vgt))
    # print("Vseg={}".format(Vseg))
    # print("VOE=", VOE)
    # print("done!")
    # print('-' * 30)
    return VOE


def HD_count(input1, input2):
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()

    if label_load.shape != predict_load.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    quality=dict()
    labelPred = sitk.GetImageFromArray(img1, isVector=False)
    labelTrue = sitk.GetImageFromArray(img2, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    # quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    # dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    # quality["dice"] = dicecomputer.GetDiceCoefficient()
    return quality["Hausdorff"]


def MSD_count(input1, input2):
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()

    if label_load.shape != predict_load.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    quality = dict()
    labelPred = sitk.GetImageFromArray(img1, isVector=False)
    labelTrue = sitk.GetImageFromArray(img2, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    # dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    # quality["dice"] = dicecomputer.GetDiceCoefficient()
    return quality["avgHausdorff"]


def dice_coef(input1, input2):
    smooth = 1.
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()


    m1 = img1.flatten()  # Flatten
    m2 = img2.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def relative_volume_diatance(input1, input2):
    input1_label_path = os.path.join(input1)
    input2_predict_path = os.path.join(input2)

    label_load = nib.load(input1_label_path)  # 读取label.nii
    predict_load = nib.load(input2_predict_path)  # 读取predict.nii

    img1 = label_load.get_fdata()
    img2 = predict_load.get_fdata()

    m1 = img1.flatten()  # Flatten  # label
    m2 = img2.flatten()  # Flatten  # predict
    return m2.sum()/m1.sum() - 1

def jac_count(input1, input2):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(input1, sitk.sitkInt16))
    label = sitk.GetArrayFromImage(sitk.ReadImage(input2, sitk.sitkInt16))

    #### 计算 dice
    zeros = np.zeros(seg.shape)  # 全0变量
    ones = np.ones(seg.shape)  # 全1变量
    tp = ((seg == ones) & (label == ones)).sum()
    fp = ((seg == zeros) & (label == ones)).sum()
    tn = ((seg == zeros) & (label == zeros)).sum()
    fn = ((seg == ones) & (label == zeros)).sum()
    jac = tp / (fp + tp + fn)
    return jac

def precision_count(input1, input2):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(input1, sitk.sitkInt16))
    label = sitk.GetArrayFromImage(sitk.ReadImage(input2, sitk.sitkInt16))

    #### 计算 dice
    zeros = np.zeros(seg.shape)  # 全0变量
    ones = np.ones(seg.shape)  # 全1变量
    tp = ((seg == ones) & (label == ones)).sum()
    fp = ((seg == zeros) & (label == ones)).sum()
    core = 0.000000000000000001
    precision = (tp + core) / (tp + fp + core)
    return precision

if __name__ == '__main__':


    #input_label_base = 'F:/Data/ON_Data/Finish/ON_Data_128x160x128_102/All/Test_Set'
    input_label_base  = "/home/AVP数据/128x160x128_102/Test_Set_FM/"

    predice_name = "/home/ON_segmentation_A222/2D_FusionNet/predict_MEnet_t1fa_fusion_mask/"

    pre_dir = os.listdir(predice_name + '/')
    input_predict_base = predice_name + '/'

    all_dice = []
    all_Iou = []

    with open( predice_name + '/' +'Dice.txt','a+') as f:
        f.writelines('epoch\n')

    with open(predice_name + '/' +'RVD.txt','a+') as f:
        f.writelines('epoch\n')

    with open(predice_name + '/' +'HD.txt','a+') as f:
        f.writelines('epoch\n')

    with open(predice_name + '/' +'ASD.txt', 'a+') as f:
        f.writelines('epoch\n')

    with open(predice_name + '/' +'jac.txt', 'a+') as f:
        f.writelines('epoch\n')

    with open(predice_name + '/' +'precision.txt', 'a+') as f:
        f.writelines('epoch\n')

    mDice = 0
    Dice = 0

    mHD = 0
    HD = 0.

    mRVD = 0
    RVD = 0.

    ASD = 0
    mASD = 0

    mjac = 0
    jac = 0.

    mprecision = 0
    precision = 0.

    input_predict_name = input_predict_base
    test_dir = os.listdir(input_label_base)


    for num in test_dir:
        input1_label_nii = input_label_base +'/'+ num +'/'+ num + '_ON-label.nii.gz'
        input2_predict_nii = predice_name + '/' + 'test_result_'+ num +'/pre_final-label.nii.gz'


        # DSC
        DICE1 = dice_coef(input1_label_nii, input2_predict_nii)
        Dice = Dice + DICE1
        DICE1 = '%.4f'%(DICE1)
        print('DSC =',DICE1)
        with open(predice_name + '/' +'Dice.txt', 'a+') as f:
            f.writelines('{0}\t'.format(DICE1))

        # RVD
        RVD1 = relative_volume_diatance(input1_label_nii, input2_predict_nii)
        RVD = RVD + RVD1
        RVD1 = '%.4f' % (RVD1 * 100) + "%"
        print('RVD =',RVD1)
        with open(predice_name + '/' +'RVD.txt', 'a+') as f:
            f.writelines('{0}\t'.format(RVD1))

        # HD
        HD1 = HD_count(input1_label_nii, input2_predict_nii)
        HD = HD + HD1
        HD1 = '%.4f' % (HD1)
        print('HD =',HD1)
        with open(predice_name + '/' +'HD.txt', 'a+') as f:
            f.writelines('{0}\t'.format(HD1))

        # ASD
        ASD1 = MSD_count(input1_label_nii, input2_predict_nii)
        ASD = ASD + ASD1
        ASD1 = '%.4f' % (ASD1)
        print('ASD =', ASD1)
        with open(predice_name + '/' +'ASD.txt', 'a+') as f:
            f.writelines('{0}\t'.format(ASD1))

        jac1 = jac_count(input1_label_nii, input2_predict_nii)
        jac = jac + jac1
        jac1 = '%.4f' % (jac1)
        print('jac =', jac1)
        with open(predice_name + '/' + 'jac.txt', 'a+') as f:
            f.writelines('{0}\t'.format(jac1))

        precision1 = precision_count(input1_label_nii, input2_predict_nii)
        precision = precision + precision1
        precision1 = '%.4f' % (precision1)
        print('precision =', precision1)
        with open(predice_name + '/' + 'precision.txt', 'a+') as f:
            f.writelines('{0}\t'.format(precision1))

    # input1_label_nii = "/home/AVP数据/cilinic_testset/ses-c01r1/new_LABEL.nii.gz"
    # input2_predict_nii = "/home/AVP Seg/2D_FusionNet/predict_clinic_2sff_fr_125_38_401_t1fapeaks_fusion_mask/test_result_ses-c01r1/pre_final-label.nii.gz"
    # # DSC
    # DICE1 = dice_coef(input1_label_nii, input2_predict_nii)
    # Dice = Dice + DICE1
    # DICE1 = '%.4f' % (DICE1)
    # print('DSC =', DICE1)
    # with open(predice_name + '/' + 'Dice.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(DICE1))
    #
    # # RVD
    # RVD1 = relative_volume_diatance(input1_label_nii, input2_predict_nii)
    # RVD = RVD + RVD1
    # RVD1 = '%.4f' % (RVD1 * 100) + "%"
    # print('RVD =', RVD1)
    # with open(predice_name + '/' + 'RVD.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(RVD1))
    #
    # # HD
    # HD1 = HD_count(input1_label_nii, input2_predict_nii)
    # HD = HD + HD1
    # HD1 = '%.4f' % (HD1)
    # print('HD =', HD1)
    # with open(predice_name + '/' + 'HD.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(HD1))
    #
    # # ASD
    # ASD1 = MSD_count(input1_label_nii, input2_predict_nii)
    # ASD = ASD + ASD1
    # ASD1 = '%.4f' % (ASD1)
    # print('ASD =', ASD1)
    # with open(predice_name + '/' + 'ASD.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(ASD1))
    #
    # jac1 = jac_count(input1_label_nii, input2_predict_nii)
    # jac = jac + jac1
    # jac1 = '%.4f' % (jac1)
    # print('jac =', jac1)
    # with open(predice_name + '/' + 'jac.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(jac1))
    #
    # precision1 = precision_count(input1_label_nii, input2_predict_nii)
    # precision = precision + precision1
    # precision1 = '%.4f' % (precision1)
    # print('precision =', precision1)
    # with open(predice_name + '/' + 'precision.txt', 'a+') as f:
    #     f.writelines('{0}\t'.format(precision1))

    mDice = Dice / 20
    mDice = '%.4f' % (mDice)
    print("mDice =", mDice)
    with open(predice_name + '/' +'Dice.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mDice))

    mASD = ASD / 20
    mASD = '%.4f' % (mASD)
    print("mASD =", mASD)
    with open(predice_name + '/' + 'ASD.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mASD))

    mHD = HD / 20
    mHD = '%.4f' % (mHD)
    print("mHD =", mHD)
    with open(predice_name + '/' + 'HD.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mHD))

    mjac = jac / 20
    mjac = '%.4f' % (mjac)
    print("mjac =", mjac)
    with open(predice_name + '/' + 'jac.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mjac))

    mprecision = precision / 20
    mprecision = '%.4f' % (mprecision)
    print("mprecision =", mprecision)
    with open(predice_name + '/' + 'precision.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mprecision))

    mRVD = RVD / 20
    mRVD = '%.4f' % (mRVD * 100) + "%"
    print("mRVD =", mRVD)
    with open(predice_name + '/' +'RVD.txt', 'a+') as f:
        f.writelines('{0}\t\n'.format(mRVD))