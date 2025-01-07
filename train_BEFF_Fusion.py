# import visdom
import config_2d
from NetModel import Unet_2d, Unet_plus_2d, Newnet
import torch, time
from torch import nn, optim
from torch.utils.data import DataLoader
from traindataset_2d import MyTrainDataset
from torchvision.transforms import transforms
from metrics_2d import dice_loss, dice
import os
import math
import setproctitle
setproctitle.setproctitle('RGVPSeg_BEFF')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
unet2d = Unet_2d.UNet2D  # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D  # U-Net++
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
flag_gpu = config_2d.FLAG_GPU

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add(inputs1, inputs2):
    intput = torch.cat([inputs1, inputs2], 1)
    return intput


# train
def train():
    global model1, model2, losses

    lr = 0.002  #
    op_lr = 0.0004  #

    model1_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_clinic_t1/t1_71epoch_32batch.pth"
    model2_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_clinic_fa/fa_89epoch_32batch.pth"
    # model3_path = "/home/ON_segmentation_A222/2D_FusionNet/outputs5_peaks/peaks_100epoch_32batch.pth"
    model4_path = "/home/ON_segmentation_A222/2D_FusionNet/weight5/t1fa_24epoch_32batch.pth"
    # 模型选择
    # 2D U-Net
    model1 = unet2d(1, 2).to(device)
    model2 = unet2d(1, 2).to(device)
    # model3 = unet2d(9, 2).to(device)
    model4 = fusionnet(4, 2).to(device)
    # model5 = fusionnet(4, 2).to(device)
    # model6 = fusionnet(6, 2).to(device)

    # 训练集选择
    ON_train_x_t1_dir = 'ON_mydata/train_clinic_data/x_t1_data/'
    ON_train_x_fa_dir = 'ON_mydata/train_clinic_data/x_fa_data/'
    ON_train_x_peaks_dir = 'ON_mydata/train_clinic_data/x_peaks_data/'
    ON_train_y_dir = 'ON_mydata/train_clinic_data/y_data/'

    # 损失函数选择

    losses1 = dice_loss()
    losses2 = torch.nn.CrossEntropyLoss()

    # 是否使用多块GPU
    if flag_gpu == 1:
        model1 = nn.DataParallel(model1).cuda()
        model2 = nn.DataParallel(model2).cuda()
        # model3 = nn.DataParallel(model3).cuda()
        model4 = nn.DataParallel(model4).cuda()
        # model5 = nn.DataParallel(model5).cuda()
        # model6 = nn.DataParallel(model6).cuda()

    train_dataset = MyTrainDataset(ON_train_x_t1_dir, ON_train_x_fa_dir, ON_train_x_peaks_dir, ON_train_y_dir,
                                   x_transform=x_transforms, z_transform=z_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model1.load_state_dict(torch.load(model1_path, map_location='cuda'))
    model2.load_state_dict(torch.load(model2_path, map_location='cuda'))
    # model3.load_state_dict(torch.load(model3_path, map_location='cuda'))
    model4.load_state_dict(torch.load(model4_path, map_location='cuda'))

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

    # optimizer1 = optim.Adam(model1.parameters(), lr=op_lr)
    #
    # t = 10  # warmup
    # T = 200  # 共有200个epoch，则用于cosine rate的一共有180个epoch
    # n_t = 0.5
    # lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t)))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda1)

    for epoch in range(0, 100):

        dt_size = len(train_dataloader.dataset)
        ## model1
        epoch_loss_t1fa = 0
        epoch_dice_t1fa = 0

        # ## model2
        # epoch_loss_t1peaks = 0
        # epoch_dice_t1peaks = 0
        #
        # ## model3
        # epoch_loss_t1fapeaks = 0
        # epoch_dice_t1fapeaks = 0

        step = 0

        optimizer4 = optim.Adam(model4.parameters(), lr=op_lr)
        # optimizer5 = optim.Adam(model5.parameters(), lr=op_lr)
        # optimizer6 = optim.Adam(model6.parameters(), lr=op_lr)

        model1.eval()
        model2.eval()
        # model3.eval()
        model4.train()
        # model5.train()
        # model6.train()

        for x_t1, x_fa, x_peaks, y in train_dataloader:

            step += 1
            inputs_t1 = x_t1.to(device)
            inputs_fa = x_fa.to(device)
            # inputs_peaks = x_peaks.to(device)
            groundtruth = y.to(device)
            # 梯度清零
            optimizer4.zero_grad()
            # optimizer5.zero_grad()
            # optimizer6.zero_grad()

            outputs_t1 = model1(inputs_t1)
            outputs_fa = model2(inputs_fa)
            # outputs_peaks = model3(inputs_peaks)
            # input1 = add(inputs_t1, inputs_peaks)
            # X = input1.shape
            # input2 = add(input1, inputs_peaks)
            # Y = input2.shape
            fusion_outputs1 = model4(outputs_t1, outputs_fa)
            fusion_predict1 = fusion_outputs1[:, 1, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            # fusion_outputs2 = model5(outputs_t1, outputs_peaks)
            # fusion_predict2 = fusion_outputs2[:, 1, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            # fusion_outputs3 = model6(torch.cat([outputs_t1, outputs_fa], dim=1), outputs_peaks)
            # fusion_predict3 = fusion_outputs3[:, 1, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            y_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]

            loss_t1fa = losses1(fusion_predict1, y_truth) + losses2(fusion_predict1, y_truth)
            # loss_t1peaks = losses1(fusion_predict2, y_truth) + losses2(fusion_predict2, y_truth)
            # loss_t1fapeaks = losses1(fusion_predict3, y_truth) + losses2(fusion_predict3, y_truth)

            label_dice_t1fa = dice(fusion_predict1, y_truth)
            # label_dice_t1peaks = dice(fusion_predict2, y_truth)
            # label_dice_t1fapeaks = dice(fusion_predict3, y_truth)

            # 反向传播
            loss_t1fa.backward(retain_graph=True)
            # loss_t1peaks.backward(retain_graph=True)
            # loss_t1fapeaks.backward(retain_graph=True)

            # 梯度更新
            optimizer4.step()
            # optimizer5.step()
            # optimizer6.step()

            epoch_loss_t1fa += float(loss_t1fa.item())
            epoch_dice_t1fa += float(label_dice_t1fa.item())
            step_loss_t1fa = loss_t1fa.item()
            step_dice_t1fa = label_dice_t1fa.item()

            # epoch_loss_t1peaks += float(loss_t1peaks.item())
            # epoch_dice_t1peaks += float(label_dice_t1peaks.item())
            # step_loss_t1peaks = loss_t1peaks.item()
            # step_dice_t1peaks = label_dice_t1peaks.item()
            #
            # epoch_loss_t1fapeaks += float(loss_t1fapeaks.item())
            # epoch_dice_t1fapeaks += float(label_dice_t1fapeaks.item())
            # step_loss_t1fapeaks = loss_t1fapeaks.item()
            # step_dice_t1fapeaks = label_dice_t1fapeaks.item()

            if step % 10 == 0:
                with open(r'loss/2DUnet_train5_clinic_BEFF_t1fa_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                   f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fa)))

                # with open(r'loss/2DUnet_train5_BEFF_t1peaks_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1peaks)))
                #
                # with open(r'loss/2DUnet_train5_BEFF_t1fapeaks_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fapeaks)))

            print("epoch:%d/%d, %d/%d, loss_BEFF_t1fa:%0.3f, label_dice_BEFF_t1fa:%0.3f, op_lr:%0.5f" % (epoch + 1,
                                                                                                         n_epochs,
                                                                                                         step * train_dataloader.batch_size,
                                                                                                         dt_size,
                                                                                                         step_loss_t1fa,
                                                                                                         step_dice_t1fa,
                                                                                                         op_lr))

            # print("epoch:%d/%d, %d/%d, loss_BEFF_t1peaks:%0.3f, label_dice_BEFF_t1peaks:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                          n_epochs,
            #                                                                                          step * train_dataloader.batch_size,
            #                                                                                          dt_size,
            #                                                                                          step_loss_t1peaks,
            #                                                                                          step_dice_t1peaks,
            #                                                                                          op_lr))
            #
            # print("epoch:%d/%d, %d/%d, loss_BEFF_t1fapeaks:%0.3f, label_dice_BEFF_t1fapeaks:%0.3f, op_lr:%0.5f" % (
            #                                                                                         epoch + 1,
            #                                                                                         n_epochs,
            #                                                                                         step * train_dataloader.batch_size,
            #                                                                                         dt_size,
            #                                                                                         step_loss_t1fapeaks,
            #                                                                                         step_dice_t1fapeaks,
            #                                                                                         op_lr))
        # scheduler.step()
        model1_path = 'outputs5_clinic_BEFF_t1fa/' + 't1fa_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model2_path = 'outputs5_BEFF_t1peaks/' + 't1peaks_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model3_path = 'outputs5_BEFF_t1fapeaks/' + 't1fapeaks_%depoch_%dbatch.pth' % (epoch + 1, batch_size)

        torch.save(model4.state_dict(), model1_path)
        # torch.save(model5.state_dict(), model2_path)
        # torch.save(model6.state_dict(), model3_path)

        train_epoch_loss_t1fa = epoch_loss_t1fa / step
        train_epoch_dice_t1fa = epoch_dice_t1fa / step

        # train_epoch_loss_t1peaks = epoch_loss_t1peaks / step
        # train_epoch_dice_t1peaks = epoch_dice_t1peaks / step
        #
        # train_epoch_loss_t1fapeaks = epoch_loss_t1fapeaks / step
        # train_epoch_dice_t1fapeaks = epoch_dice_t1fapeaks / step

        with open(r'loss/2DUnet_val5_clinic_BEFF_t1fa_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
            f.writelines('epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_dice_t1fa)))

        # with open(r'loss/2DUnet_val5_BEFF_t1peaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines('epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_dice_t1peaks)))
        #
        # with open(r'loss/2DUnet_val5_BEFF_t1fapeaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines('epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_dice_t1fapeaks)))

        print("epoch:%d, train_loss_BEFF_t1fa:%0.3f, train_dice_BEFF_t1fa:%0.3f" % (epoch + 1,
                                                                                    train_epoch_loss_t1fa,
                                                                                    train_epoch_dice_t1fa))

        # print("epoch:%d, train_loss_BEFF_t1peaks:%0.3f, train_dice_BEFF_t1peaks:%0.3f" % (epoch + 1,
        #                                                                             train_epoch_loss_t1peaks,
        #                                                                             train_epoch_dice_t1peaks))
        #
        # print("epoch:%d, train_loss_BEFF_t1fapeaks:%0.3f, train_dice_BEFF_t1fapeaks:%0.3f" % (epoch + 1,
        #                                                                             train_epoch_loss_t1fapeaks,
        #                                                                             train_epoch_dice_t1fapeaks))

    print('-' * 30)


### 训练数据融合模型
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
    if 'outputs5_clinic_BEFF_t1fa' not in os.listdir(os.curdir):
        os.mkdir('outputs5_clinic_BEFF_t1fa')
    # if 'outputs5_BEFF_t1peaks' not in os.listdir(os.curdir):
    #     os.mkdir('outputs5_BEFF_t1peaks')
    # if 'outputs5_BEFF_t1fapeaks' not in os.listdir(os.curdir):
    #     os.mkdir('outputs5_BEFF_t1fapeaks')

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
