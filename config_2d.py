import argparse
VOLUME_ROWS = 128
VOLUME_COLS = 160
VOLUME_DEPS = 128

NUM_CLASSES = 2

PATCH_SIZE_W = 128 # 裁剪的尺寸和输入网络的图像尺寸
PATCH_SIZE_H = 160

BATCH_SIZE = 32 # 一次输入多少图像进入网络
NUM_EPOCHS = 100



TRAIN_EXTRACTION_STEP = 12                 # 创建训练集提取的步长
TEST_EXTRACTION_STEP = 1      # 创建测试集提取的步长

# 路径设置
COM_CHOOSE = 3
if COM_CHOOSE ==1:  # lsq
    train_imgs_path = 'F:/Data/ON_Data/Finish/ON_Data_128x160x128_102/All/Train_Set'
    val_imgs_path = 'F:/Data/ON_Data/Finish/ON_Data_128x160x128_102/All/Val_Set'
    test_imgs_path =  'F:/Data/ON_Data/Finish/ON_Data_128x160x128_102/All/Test_Set'
if COM_CHOOSE == 2: # gwl
    train_imgs_path = 'E:\LSQ\ON_data\ON_Data_128x160x128_102\All\Train_Set'    # 128x160x128
    val_imgs_path = 'E:\LSQ\ON_data\ON_Data_128x160x128_102\All\Val_Set'
    test_imgs_path =  'E:\LSQ\ON_data\ON_Data_128x160x128_102\All\Test_Set'     # 128x160x128
if COM_CHOOSE == 3: # A222
    train_imgs_path = '/home/AVP数据/128x160x128_102/Train_Set_FM/'
    # val_imgs_path = "/home/AVP数据/128x160x128_102/Val_Set/"
    test_imgs_path = '/home/AVP数据/128x160x128_102/Test_Set_FM/'




# 是否选用多块GPU
FLAG_GPU = 1