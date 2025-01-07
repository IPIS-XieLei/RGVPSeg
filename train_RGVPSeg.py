import os
import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d, Newnet,SFFnet
import time
from torch import optim
from torch.utils.data import DataLoader
from traindataset_2d import MyTrainDataset
from torchvision.transforms import transforms
from metrics_2d import dice_loss, dice
import torch
import torch.nn as nn
import math
import setproctitle
setproctitle.setproctitle("RGVPSeg")
# os.environ['CUDA_VISIBLE_DEVICES']='1'

unet2d = Unet_2d.UNet2D  # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D  # U-Net++
multiresunet2d = MultiResUnet_2d.MultiResUnet2D  # MultiRes U-Net
ournet2d = MultiResUnet_plus_2d.MultiResUnetPlus2D  # MultiRes U-Net++
fusionnet = Newnet.DatafusionNet

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path
t1sfffaFuseUNet1=SFFnet.t1sfffaFuseUNet1
t1sffpeaksFuseUNet2=SFFnet.t1sffpeaksFuseUNet2
flag_gpu = config_2d.FLAG_GPU

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# train
def train():
    global model1, model2, model3

    op_lr = 0.0004  #

    # 模型选择
    # sanet
    model1 = t1sfffaFuseUNet1(1, 1).to(device)
    model2 = t1sffpeaksFuseUNet2(1, 9).to(device)
    # model3 = t1safapeaksF useUNet3(2, 9).to(device)
    model4 = fusionnet(4, 2).to(device)
    # model = entire_net().to(device)
    #### 训练集选择
    ON_train_x_t1_dir = 'ON_mydata/train1_clinic_data/x_t1_data/'
    ON_train_x_fa_dir = 'ON_mydata/train1_clinic_data/x_fa_data/'
    ON_train_x_peaks_dir = 'ON_mydata/train1_clinic_data/x_peaks_data/'
    ON_train_y_dir = 'ON_mydata/train1_clinic_data/y_data/'

    # 损失函数选择
    losses1 = dice_loss()
    losses2 = torch.nn.CrossEntropyLoss()
    # 是否使用多块GPU
    if flag_gpu == 1:
        model1 = nn.DataParallel(model1).cuda()
        model2 = nn.DataParallel(model2).cuda()
        # model3 = nn.DataParallel(model3).cuda()
        model4 = nn.DataParallel(model4).cuda()
        # model = nn.DataParallel(model).cuda()

    train_dataset = MyTrainDataset(ON_train_x_t1_dir, ON_train_x_fa_dir, ON_train_x_peaks_dir, ON_train_y_dir,
                                   x_transform=x_transforms, z_transform=z_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ###----------------------
    #### start train
    print('-' * 30)
    print('Training start...')
    print('-' * 30)
    print('patch size   : ', patch_size_w, 'x', patch_size_h)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('learning rate: ', op_lr)
    print('-' * 30)

    model1.load_state_dict(torch.load(r"outputs2_clinic_sff_fr_t1fa/t1fa_74epoch_32batch.pth",map_location='cuda'))
    model2.load_state_dict(torch.load(r"outputs2_clinic_sff_fr_t1peaks/t1peaks_78epoch_32batch.pth",map_location='cuda'))
    # model3.load_state_dict(torch.load(r"/home/AVP Seg/2D_FusionNet/outputs_sa_fr_diceBCE_t1fapeaks/t1fapeaks_59epoch_32batch-net3.pth",map_location='cuda'))
    model4.load_state_dict(torch.load("outputs2_clinic_2sff_fr_74_78_fusion_t1fapeaks/t1fapeaks_11epoch_32batch",map_location='cuda'))
    # model.load_state_dict(torch.load("/home/AVP Seg/2D_FusionNet/outputs_entire_net_diceBCE_t1fapeaks/t1fapeaks_6epoch_32batch", map_location='cuda'))

    for epoch in range(0, 100):

        dt_size = len(train_dataloader.dataset)

        # # model1
        # epoch_loss_t1fa = 0
        # epoch_dice_t1fa = 0
        # loss_t1fa = 0
        #
        # # model2
        # epoch_loss_t1peaks = 0
        # epoch_dice_t1peaks = 0
        # loss_t1peaks = 0

        # model3
        epoch_loss_t1fapeaks = 0
        epoch_dice_t1fapeaks = 0
        loss_t1fapeaks = 0

        step = 0

        # optimizer1 = optim.Adam(model1.parameters(), lr=op_lr)
        # optimizer2 = optim.Adam(model2.parameters(), lr=op_lr)
        # optimizer3 = optim.Adam(model3.parameters(), lr=op_lr)
        optimizer4 = optim.Adam(model4.parameters(), lr=op_lr)
        # optimizer = optim.Adam(model.parameters(), lr=op_lr)

        model1.eval()
        model2.eval()
        # model3.train()
        model4.train()
        # model.train()

        for x_t1, x_fa, x_peaks, y in train_dataloader:

            step += 1
            inputs_t1 = x_t1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            inputs_fa = x_fa.to(device)
            inputs_peaks = x_peaks.to(device)
            groundtruth = y.to(device)
            # 梯度清零
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # optimizer3.zero_grad()
            optimizer4.zero_grad()

            outputs_t1fa = model1(inputs_t1, inputs_fa)
            outputs_t1peaks = model2(inputs_t1, inputs_peaks)
            outputs_t1fapeaks = model4(outputs_t1fa, outputs_t1peaks)
            # outputs_t1fapeaks = model(inputs_t1, inputs_fa, inputs_peaks)

            # z = outputs_t1peaks.shape

            # label_t1fa_predict = outputs_t1fa[:, 1, :, :].squeeze()
            # label_t1peaks_predict = outputs_t1peaks[:, 1, :, :].squeeze()
            label_t1fapeaks_predict = outputs_t1fapeaks[:, 1, :, :].squeeze()

            # t1 = label_peaks_predict.shape

            label_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]
            # t2 = label_truth.shape

            # loss_t1fa = losses1(label_t1fa_predict, label_truth) + losses2(label_t1fa_predict, label_truth)
            # loss_t1peaks = losses1(label_t1peaks_predict, label_truth) + losses2(label_t1peaks_predict, label_truth)
            loss_t1fapeaks = losses1(label_t1fapeaks_predict, label_truth) + losses2(label_t1fapeaks_predict, label_truth)


            # label_t1fa_dice = dice(label_t1fa_predict, label_truth)
            # label_t1peaks_dice = dice(label_t1peaks_predict, label_truth)
            label_t1fapeaks_dice = dice(label_t1fapeaks_predict, label_truth)

            # 反向传播
            # loss_t1fa.backward()
            # loss_t1peaks.backward()
            loss_t1fapeaks.backward()


            # 梯度更新
            # optimizer1.step()
            # optimizer2.step()
            # optimizer3.step()
            optimizer4.step()
            # optimizer.step()

            # epoch_loss_t1fa += float(loss_t1fa.item())
            # epoch_dice_t1fa += float(label_t1fa_dice.item())
            # step_loss_t1fa = loss_t1fa.item()
            # step_dice_t1fa = label_t1fa_dice.item()
            #
            # epoch_loss_t1peaks += float(loss_t1peaks.item())
            # epoch_dice_t1peaks += float(label_t1peaks_dice.item())
            # step_loss_t1peaks = loss_t1peaks.item()
            # step_dice_t1peaks = label_t1peaks_dice.item()

            epoch_loss_t1fapeaks += float(loss_t1fapeaks.item())
            epoch_dice_t1fapeaks += float(label_t1fapeaks_dice.item())
            step_loss_t1fapeaks = loss_t1fapeaks.item()
            step_dice_t1fapeaks = label_t1fapeaks_dice.item()

            if step % 10 == 0:
                # with open(r'loss/sffnet_train2_clinic_t1fa_' + str(batch_size) + 'batch_step_loss.txt',
                #           'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fa)))
                #
                # with open(r'loss/sffnet_train2_clinic_t1peaks_' + str(batch_size) + 'batch_step_loss.txt',
                #           'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1peaks)))

                with open(r'loss/sffnet_train2_clinic_t1fapeaks_' + str(batch_size) + 'batch_step_loss.txt',
                          'a+') as f:
                    f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fapeaks)))

            # print("epoch:%d/%d, %d/%d, loss_t1fa:%0.3f, label_dice_t1fa:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                              n_epochs,
            #                                                                                              step * train_dataloader.batch_size,
            #                                                                                              dt_size,
            #                                                                                              step_loss_t1fa,
            #                                                                                              step_dice_t1fa,
            #                                                                                              op_lr))
            #
            # print("epoch:%d/%d, %d/%d, loss_t1peaks:%0.3f, label_dice_t1peaks:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                              n_epochs,
            #                                                                                              step * train_dataloader.batch_size,
            #                                                                                              dt_size,
            #                                                                                              step_loss_t1peaks,
            #                                                                                              step_dice_t1peaks,
            #                                                                                              op_lr))

            print("epoch:%d/%d, %d/%d, loss_t1fapeaks:%0.3f, label_dice_t1fapeaks:%0.3f, op_lr:%0.5f" % (epoch + 1,
                                                                                                                     n_epochs,
                                                                                                                     step * train_dataloader.batch_size,
                                                                                                                     dt_size,
                                                                                                                     step_loss_t1fapeaks,
                                                                                                                     step_dice_t1fapeaks,
                                                                                                                     op_lr))

        # model1_path = 'outputs2_clinic_sff_fr_t1fa/' + 't1fa_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model2_path = 'outputs2_clinic_sff_fr_t1peaks/' + 't1peaks_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model3_path = 'outputs_sa_fr_diceBCE_t1fapeaks/' + 't1fapeaks_%depoch_%dbatch-net3.pth' % (epoch + 1, batch_size)
        model4_path = 'outputs2_clinic_2sff_fr_74_78_fusion_t1fapeaks/' + 't1fapeaks_%depoch_%dbatch' % (epoch + 1, batch_size)
        # model_path = 'outputs_2sff_fr_125_38_fusion_diceBCE_t1fapeaks/' + 't1fapeaks_%depoch_%dbatch' % (epoch + 1, batch_size)
        # torch.save(model1.state_dict(), model1_path)
        # torch.save(model2.state_dict(), model2_path)
        # torch.save(model3.state_dict(), model3_path)
        torch.save(model4.state_dict(), model4_path)
        # torch.save(model.state_dict(), model_path)

        # train_epoch_loss_t1fa = epoch_loss_t1fa / step
        # train_epoch_dice_t1fa = epoch_dice_t1fa / step
        #
        # train_epoch_loss_t1peaks = epoch_loss_t1peaks / step
        # train_epoch_dice_t1peaks = epoch_dice_t1peaks / step

        train_epoch_loss_t1fapeaks = epoch_loss_t1fapeaks / step
        train_epoch_dice_t1fapeaks = epoch_dice_t1fapeaks / step

        # with open(r'loss/sffnet_val2_clinic_t1fa_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1}\t{2} \n'.format(str(epoch + 1), str(train_epoch_loss_t1fa),
        #                                        str(train_epoch_dice_t1fa)))
        #
        # with open(r'loss/sffnet_val2_clinic_t1peaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1}\t{2} \n'.format(str(epoch + 1), str(train_epoch_loss_t1peaks),
        #                                        str(train_epoch_dice_t1peaks)))

        with open(r'loss/sffnet_val2_clinic_t1fapeaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
            f.writelines(
                'epoch{0}\t{1}\t{2} \n'.format(str(epoch + 1), str(train_epoch_loss_t1fapeaks),
                                               str(train_epoch_dice_t1fapeaks)))

        # print("epoch:%d, train_loss_t1fa:%0.3f, train_dice_t1fa:%0.3f" % (
        #     epoch + 1, train_epoch_loss_t1fa, train_epoch_dice_t1fa))
        #
        # print("epoch:%d, train_loss_t1peaks:%0.3f, train_dice_t1peaks:%0.3f" % (
        #     epoch + 1, train_epoch_loss_t1peaks, train_epoch_dice_t1peaks))

        print("epoch:%d, train_loss_t1fapeaks:%0.3f, train_dice_t1fapeaks:%0.3f" % (
            epoch + 1, train_epoch_loss_t1fapeaks, train_epoch_dice_t1fapeaks))

    print('-' * 30)


### 训练各自模型
if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    z_transforms = transforms.ToTensor()
    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # 模型保存
    # if 'outputs2_clinic_sff_fr_t1fa' not in os.listdir(os.curdir):
    #     os.mkdir('outputs2_clinic_sff_fr_t1fa')
    #
    # if 'outputs2_clinic_sff_fr_t1peaks' not in os.listdir(os.curdir):
    #     os.mkdir('outputs2_clinic_sff_fr_t1peaks')

    if 'outputs2_clinic_2sff_fr_74_78_fusion_t1fapeaks' not in os.listdir(os.curdir):
        os.mkdir('outputs2_clinic_2sff_fr_74_78_fusion_t1fapeaks')

    # loss保存
    if 'loss' not in os.listdir(os.curdir):
        os.mkdir('loss')

    ### train test ###
    start_time = time.time()
    train()
    end_time = time.time()
    print("2D train time is {:.3f} mins".format((end_time - start_time) / 60.0))
    print('-' * 30)
    print('patch size   : ', patch_size_w, 'x', patch_size_h)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('-' * 30)
    print("done")
