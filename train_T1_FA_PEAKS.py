# import visdom
import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d
import torch, time
from torch import nn, optim
from torch.utils.data import DataLoader
from traindataset_2d import MyTrainDataset
from torchvision.transforms import transforms
from metrics_2d import dice_loss, dice
import os
import math
import setproctitle


setproctitle.setproctitle("RGVPSeg_TTFDP")
os.environ['CUDA_VISIBLE_DEVICES']='1'
unet2d = Unet_2d.UNet2D  # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D  # U-Net++
multiresunet2d = MultiResUnet_2d.MultiResUnet2D  # MultiRes U-Net
ournet2d = MultiResUnet_plus_2d.MultiResUnetPlus2D  # MultiRes U-Net++

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


# train
def train():
    global model1, model2, model3

    lr = 0.002  #
    op_lr = 0.0004  #

    # 模型选择
    # 2D U-Net
    # model1 = unet2d(1, 2).to(device)
    model2 = unet2d(1, 2).to(device)
    # model3 = unet2d(1, 2).to(device)
    model4 = unet2d(3, 2).to(device)
    # model5 = unet2d(9, 2).to(device)


    #### 训练集选择
    ON_train_x_t1_dir = 'ON_mydata/train_data/x_t1_data/'
    ON_train_x_t2_dir = 'ON_mydata/train_data/x_t2_data/'
    ON_train_x_fa_dir = 'ON_mydata/train_data/x_fa_data/'
    ON_train_x_dec_dir = 'ON_mydata/train_data/x_dec_data/'
    ON_train_x_peaks_dir = 'ON_mydata/train_data/x_peaks_data/'
    ON_train_y_dir = 'ON_mydata/train_data/y_data/'

    ### 验证集选择
    # ON_val_x_t1_dir = 'ON_mydata/val_data/x_t1_data/'
    # ON_val_x_fa_dir = 'ON_mydata/val_data/x_fa_data/'
    # ON_val_y_dir    = 'ON_mydata/val_data/y_data/'

    # 损失函数选择
    losses1 = dice_loss()
    losses2 = torch.nn.CrossEntropyLoss()

    # 是否使用多块GPU
    if flag_gpu == 1:
        # model1 = nn.DataParallel(model1).cuda()
        model2 = nn.DataParallel(model2).cuda()
        # model3 = nn.DataParallel(model3).cuda()
        model4 = nn.DataParallel(model4).cuda()
        # model5 = nn.DataParallel(model5).cuda()

    train_dataset = MyTrainDataset(ON_train_x_t1_dir, ON_train_x_t2_dir, ON_train_x_fa_dir, ON_train_x_dec_dir,
                                   ON_train_x_peaks_dir, ON_train_y_dir,
                                   x_transform=x_transforms, z_transform=z_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # val_dataset = MyTrainDataset(ON_val_x_t1_dir, ON_val_x_fa_dir, ON_val_y_dir, x_transform=x_transforms, y_transform=y_transforms)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

    # model1.load_state_dict(torch.load(r"/home/ON_segmentation_A222/2D_FusionNet/weight5/t1_53epoch_32batch.pth", map_location='cuda'))
    # model2.load_state_dict(torch.load(r"/home/ON_segmentation_A222/2D_FusionNet/weight5/fa_64epoch_32batch.pth", map_location='cuda'))
    # model3.load_state_dict(torch.load(r"outputs_peaks/peaks_20epoch_64batch.pth", map_location='cuda'))

    for epoch in range(0, 100):

        dt_size = len(train_dataloader.dataset)

        ## model1
        # epoch_loss_t1 = 0
        # epoch_dice_t1 = 0
        # loss_t1 = 0

        ## model2
        epoch_loss_t2 = 0
        epoch_dice_t2 = 0
        loss_t2 = 0

        ## model3
        # epoch_loss_fa = 0
        # epoch_dice_fa = 0
        # loss_fa = 0

        ## model4
        epoch_loss_dec = 0
        epoch_dice_dec = 0
        loss_dec = 0

        # ## model5
        # epoch_loss_peaks = 0
        # epoch_dice_peaks = 0
        # loss_peaks = 0

        step = 0

        # optimizer1 = optim.Adam(model1.parameters(), lr=op_lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=op_lr)
        # optimizer3 = optim.Adam(model3.parameters(), lr=op_lr)
        optimizer4 = optim.Adam(model4.parameters(), lr=op_lr)
        # optimizer5 = optim.Adam(model5.parameters(), lr=op_lr)

        # model1.train()
        model2.train()
        # model3.train()
        model4.train()
        # model5.train()

        for x_t1, x_t2, x_fa, x_dec, x_peaks, y in train_dataloader:

            step += 1
            # inputs_t1 = x_t1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            inputs_t2 = x_t2.to(device)
            # inputs_fa = x_fa.to(device)
            inputs_dec = x_dec.to(device)
            # inputs_peaks = x_peaks.to(device)
            groundtruth = y.to(device)
            # 梯度清零
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            optimizer4.zero_grad()
            # optimizer5.zero_grad()

            # outputs_t1 = model1(inputs_t1)  # tensor[batch_size, 2, 144, 144]
            outputs_t2 = model2(inputs_t2)
            # outputs_fa = model3(inputs_fa)
            outputs_dec = model4(inputs_dec)
            # outputs_peaks = model3(inputs_peaks)

            # label_t1_predict = outputs_t1[:, 1, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]
            label_t2_predict = outputs_t2[:, 1, :, :].squeeze()
            # label_fa_predict = outputs_fa[:, 1, :, :].squeeze()
            label_dec_predict = outputs_dec[:, 1, :, :].squeeze()
            # label_peaks_predict = outputs_peaks[:, 1, :, :].squeeze()

            label_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]

            # loss_t1 = losses1(label_t1_predict, label_truth) + losses2(label_t1_predict, label_truth)
            loss_t2 = losses1(label_t2_predict, label_truth) + losses2(label_t2_predict, label_truth)
            # loss_fa = losses1(label_fa_predict, label_truth) + losses2(label_fa_predict, label_truth)
            loss_dec = losses1(label_dec_predict, label_truth) + losses2(label_dec_predict, label_truth)
            # loss_peaks = losses1(label_peaks_predict, label_truth)+losses2(label_peaks_predict, label_truth)

            # label_t1_dice = dice(label_t1_predict, label_truth)
            label_t2_dice = dice(label_t2_predict, label_truth)
            # label_fa_dice = dice(label_fa_predict, label_truth)
            label_dec_dice = dice(label_dec_predict, label_truth)
            # label_peaks_dice = dice(label_peaks_predict, label_truth)

            # 反向传播
            # loss_t1.backward()
            loss_t2.backward()
            # loss_fa.backward()
            loss_dec.backward()
            # loss_peaks.backward()

            # 梯度更新
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()
            optimizer4.step()
            # optimizer5.step()

            # epoch_loss_t1 += float(loss_t1.item())
            # epoch_dice_t1 += float(label_t1_dice.item())
            # step_loss_t1 = loss_t1.item()
            # step_dice_t1 = label_t1_dice.item()

            epoch_loss_t2 += float(loss_t2.item())
            epoch_dice_t2 += float(label_t2_dice.item())
            step_loss_t2 = loss_t2.item()
            step_dice_t2 = label_t2_dice.item()

            # epoch_loss_fa += float(loss_fa.item())
            # epoch_dice_fa += float(label_fa_dice.item())
            # step_loss_fa = loss_fa.item()
            # step_dice_fa = label_fa_dice.item()

            epoch_loss_dec += float(loss_dec.item())
            epoch_dice_dec += float(label_dec_dice.item())
            step_loss_dec = loss_dec.item()
            step_dice_dec = label_dec_dice.item()
            #
            # epoch_loss_peaks += float(loss_peaks.item())
            # epoch_dice_peaks += float(label_peaks_dice.item())
            # step_loss_peaks = loss_peaks.item()
            # step_dice_peaks = label_peaks_dice.item()

            if step % 10 == 0:
                # with open(r'loss/2DUnet_train_t1_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1)))

                with open(r'loss/2DUnet_train_t2_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                    f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t2)))

                # with open(r'loss/2DUnet_train_fa_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_fa)))

                with open(r'loss/2DUnet_train_dec_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                    f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_dec)))
                #
                # with open(r'loss/2DUnet_train_peaks_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_peaks)))

            # print("epoch:%d/%d, %d/%d, loss_t1:%0.3f, label_dice_t1:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                n_epochs,
            #                                                                                step * train_dataloader.batch_size,
            #                                                                                dt_size,
            #                                                                                step_loss_t1,
            #                                                                                step_dice_t1,
            #                                                                                op_lr))

            print("epoch:%d/%d, %d/%d, loss_t2:%0.3f, label_dice_t2:%0.3f, op_lr:%0.5f" % (epoch + 1,
                                                                                           n_epochs,
                                                                                           step * train_dataloader.batch_size,
                                                                                           dt_size,
                                                                                           step_loss_t2,
                                                                                           step_dice_t2,
                                                                                           op_lr))

            # print("epoch:%d/%d, %d/%d, loss_fa:%0.3f, label_dice_fa:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                n_epochs,
            #                                                                                step * train_dataloader.batch_size,
            #                                                                                dt_size,
            #                                                                                step_loss_fa,
            #                                                                                step_dice_fa,
            #                                                                                op_lr))

            print("epoch:%d/%d, %d/%d, loss_dec:%0.3f, label_dice_dec:%0.3f, op_lr:%0.5f" % (epoch + 1,
                                                                                           n_epochs,
                                                                                           step * train_dataloader.batch_size,
                                                                                           dt_size,
                                                                                           step_loss_dec,
                                                                                           step_dice_dec,
                                                                                           op_lr))

            # print("epoch:%d/%d, %d/%d, loss_peaks:%0.3f, label_dice_peaks:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                n_epochs,
            #                                                                                step * train_dataloader.batch_size,
            #                                                                                dt_size,
            #                                                                                step_loss_peaks,
            #                                                                                step_dice_peaks,
            #                                                                                op_lr))

        # scheduler1.step()
        # scheduler2.step()
        # scheduler3.step()
        # model1_path = 'outputs_t1/' + 't1_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        model2_path = 'outputs_t2/' + 't2_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model3_path = 'outputs_fa/' + 'fa_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        model4_path = 'outputs_dec/' + 'dec_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model5_path = 'outputs_peaks/' + 'peaks_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # torch.save(model1.state_dict(), model1_path)
        torch.save(model2.state_dict(), model2_path)
        # torch.save(model3.state_dict(), model3_path)
        torch.save(model4.state_dict(), model4_path)
        # torch.save(model5.state_dict(), model5_path)

        # train_epoch_loss_t1 = epoch_loss_t1 / step
        # train_epoch_dice_t1 = epoch_dice_t1 / step

        train_epoch_loss_t2 = epoch_loss_t2 / step
        train_epoch_dice_t2 = epoch_dice_t2 / step

        # train_epoch_loss_fa = epoch_loss_fa / step
        # train_epoch_dice_fa = epoch_dice_fa / step

        train_epoch_loss_dec = epoch_loss_dec / step
        train_epoch_dice_dec = epoch_dice_dec / step
        #
        # train_epoch_loss_peaks = epoch_loss_peaks / step
        # train_epoch_dice_peaks = epoch_dice_peaks / step

        # with open(r'loss/2DUnet_val_t1_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_t1)))
        with open(r'loss/2DUnet_val_t2_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
            f.writelines(
                'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_t2)))

        # with open(r'loss/2DUnet_val_fa_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_fa)))

        with open(r'loss/2DUnet_val_dec_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
            f.writelines(
                'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_dec)))
        #
        # with open(r'loss/2DUnet_val5_peaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_peaks)))

        # print("epoch:%d, train_loss_t1:%0.3f, train_dice_t1:%0.3f" % (
        # epoch + 1, train_epoch_loss_t1, train_epoch_dice_t1))
        print("epoch:%d, train_loss_t2:%0.3f, train_dice_t2:%0.3f" % (
            epoch + 1, train_epoch_loss_t2, train_epoch_dice_t2))
        # print("epoch:%d, train_loss_fa:%0.3f, train_dice_fa:%0.3f" % (
        # epoch + 1, train_epoch_loss_fa, train_epoch_dice_fa))
        print("epoch:%d, train_loss_dec:%0.3f, train_dice_dec:%0.3f" % (
            epoch + 1, train_epoch_loss_dec, train_epoch_dice_dec))
        # print("epoch:%d, train_loss_peaks:%0.3f, train_dice_peaks:%0.3f" % (epoch + 1, train_epoch_loss_peaks, train_epoch_dice_peaks))

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
    # if 'outputs_t1' not in os.listdir(os.curdir):
    #     os.mkdir('outputs_t1')
    if 'outputs_t2' not in os.listdir(os.curdir):
        os.mkdir('outputs_t2')
    # if 'outputs_fa' not in os.listdir(os.curdir):
    #     os.mkdir('outputs_fa')
    if 'outputs_dec' not in os.listdir(os.curdir):
        os.mkdir('outputs_dec')
    # if 'outputs_peaks' not in os.listdir(os.curdir):
    #     os.mkdir('outputs_peaks')

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
