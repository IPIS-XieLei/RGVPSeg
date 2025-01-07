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
import setproctitle
setproctitle.setproctitle('RGVPSeg_FECC')
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
    # model1 = unet2d(2, 2).to(device)
    model2 = unet2d(4, 2).to(device)
    # model3 = unet2d(11, 2).to(device)

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

    train_dataset = MyTrainDataset(ON_train_x_t1_dir,ON_train_x_t2_dir, ON_train_x_fa_dir,ON_train_x_dec_dir, ON_train_x_peaks_dir, ON_train_y_dir,
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

    # model1.load_state_dict(torch.load(r"/home/ON_segmentation_A222/2D_FusionNet/weight5/t1fa_100epoch_32batch.pth", map_location='cuda'))
    # model2.load_state_dict(torch.load(r"outputs_fa/fa_20epoch_64batch.pth", map_location='cuda'))
    # model3.load_state_dict(torch.load(r"/home/ON_segmentation_A222/2D_FusionNet/outputs2_FECC_t1fapeaks/t1fapeaks_86epoch_32batch.pth", map_location='cuda'))

    for epoch in range(0, 100):

        dt_size = len(train_dataloader.dataset)

        ## model1
        # epoch_loss_t1t2 = 0
        # epoch_dice_t1t2 = 0

        # ## model2
        epoch_loss_fadecpeaks = 0
        epoch_dice_fadecpeaks = 0
        #
        # ## model3
        # epoch_loss_t1fapeaks = 0
        # epoch_dice_t1fapeaks = 0

        step = 0

        # optimizer1 = optim.Adam(model1.parameters(), lr=op_lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=op_lr)
        # optimizer3 = optim.Adam(model3.parameters(), lr=op_lr)

        # model1.train()
        model2.train()
        # model3.train()

        for x_t1,x_t2, x_fa,x_dec, x_peaks, y in train_dataloader:

            step += 1
            inputs_t1 = x_t1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            # inputs_t2 = x_t2.to(device)
            # inputs_fa = x_fa.to(device)
            inputs_dec = x_dec.to(device)
            # inputs_peaks = x_peaks.to(device)
            groundtruth = y.to(device)
            # 梯度清零
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()

            # input_t1t2 = torch.cat((inputs_t1, inputs_t2), dim=1)
            input_fadecpeaks = torch.cat((inputs_t1, inputs_dec), dim=1)
            # input_t1fapeaks = torch.cat((torch.cat((inputs_t1, inputs_fa), dim=1), inputs_peaks), dim=1)

            # outputs_t1t2 = model1(input_t1t2)  # tensor[batch_size, 2, 144, 144]
            outputs_fadecpeaks = model2(input_fadecpeaks)
            # outputs_t1fapeaks = model3(input_t1fapeaks)

            # label_t1t2_predict = outputs_t1t2[:, 1, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]
            label_fadecpeaks_predict = outputs_fadecpeaks[:, 1, :, :].squeeze()
            # label_t1fapeaks_predict = outputs_t1fapeaks[:, 1, :, :].squeeze()

            label_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]

            # loss_t1t2 = losses1(label_t1t2_predict, label_truth) + losses2(label_t1t2_predict, label_truth)
            loss_fadecpeaks = losses1(label_fadecpeaks_predict, label_truth)+losses2(label_fadecpeaks_predict, label_truth)
            # loss_t1fapeaks = losses1(label_t1fapeaks_predict, label_truth)+losses2(label_t1fapeaks_predict, label_truth)

            # label_t1t2_dice = dice(label_t1t2_predict, label_truth)
            label_fadecpeaks_dice = dice(label_fadecpeaks_predict, label_truth)
            # label_t1fapeaks_dice = dice(label_t1fapeaks_predict, label_truth)

            # 反向传播
            # loss_t1t2.backward()
            loss_fadecpeaks.backward()
            # loss_t1fapeaks.backward()

            # 梯度更新
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()

            # epoch_loss_t1t2 += float(loss_t1t2.item())
            # epoch_dice_t1t2 += float(label_t1t2_dice.item())
            # step_loss_t1t2 = loss_t1t2.item()
            # step_dice_t1t2 = label_t1t2_dice.item()

            epoch_loss_fadecpeaks += float(loss_fadecpeaks.item())
            epoch_dice_fadecpeaks += float(label_fadecpeaks_dice.item())
            step_loss_fadecpeaks = loss_fadecpeaks.item()
            step_dice_fadecpeaks = label_fadecpeaks_dice.item()
            #
            # epoch_loss_t1fapeaks += float(loss_t1fapeaks.item())
            # epoch_dice_t1fapeaks += float(label_t1fapeaks_dice.item())
            # step_loss_t1fapeaks = loss_t1fapeaks.item()
            # step_dice_t1fapeaks = label_t1fapeaks_dice.item()

            if step % 10 == 0:
                # with open(r'loss/2DUnet_train_FECC_t1t2_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1t2)))

                with open(r'loss/2DUnet_train_FECC_t1dec_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                    f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_fadecpeaks)))
                #
                # with open(r'loss/2DUnet_train5_FECC_t1fapeaks_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
                #     f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fapeaks)))

            # print("epoch:%d/%d, %d/%d, loss_t1t2:%0.3f, label_dice_t1t2:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                    n_epochs,
            #                                                                                    step * train_dataloader.batch_size,
            #                                                                                    dt_size,
            #                                                                                    step_loss_t1t2,
            #                                                                                    step_dice_t1t2,
            #                                                                                    op_lr))
            print("epoch:%d/%d, %d/%d, loss_t1dec:%0.3f, label_dice_t1dec:%0.3f, op_lr:%0.5f" % (epoch + 1,
                                                                                           n_epochs,
                                                                                           step * train_dataloader.batch_size,
                                                                                           dt_size,
                                                                                           step_loss_fadecpeaks,
                                                                                           step_dice_fadecpeaks,
                                                                                           op_lr))
            # print("epoch:%d/%d, %d/%d, loss_t1fapeaks:%0.3f, label_dice_t1fapeaks:%0.3f, op_lr:%0.5f" % (epoch + 1,
            #                                                                                n_epochs,
            #                                                                                step * train_dataloader.batch_size,
            #                                                                                dt_size,
            #                                                                                step_loss_t1fapeaks,
            #                                                                                step_dice_t1fapeaks,
            #                                                                                op_lr))

        # model1_path = 'outputs_FECC_t1t2/' + 't1t2_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        model2_path = 'outputs_FECC_t1dec/' + 't1dec_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # model3_path = 'outputs5_FECC_t1fapeaks/' + 't1fapeaks_%depoch_%dbatch.pth' % (epoch + 1, batch_size)
        # torch.save(model1.state_dict(), model1_path)
        torch.save(model2.state_dict(), model2_path)
        # torch.save(model3.state_dict(), model3_path)

        # train_epoch_loss_t1t2 = epoch_loss_t1t2 / step
        # train_epoch_dice_t1t2 = epoch_dice_t1t2 / step

        train_epoch_loss_fadecpeaks = epoch_loss_fadecpeaks / step
        train_epoch_dice_fadecpeaks = epoch_dice_fadecpeaks / step
        #
        # train_epoch_loss_t1fapeaks = epoch_loss_t1fapeaks / step
        # train_epoch_dice_t1fapeaks = epoch_dice_t1fapeaks / step

        # with open(r'loss/2DUnet_val_FECC_t1t2_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1}\t{2} \n'.format(str(epoch + 1), str(train_epoch_loss_t1t2), str(train_epoch_dice_t1t2)))

        with open(r'loss/2DUnet_val_FECC_t1dec_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
            f.writelines(
                'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_fadecpeaks)))
        #
        # with open(r'loss/2DUnet_val5_FECC_t1fapeaks_' + str(batch_size) + 'batch_epoch_loss.txt', 'a+') as f:
        #     f.writelines(
        #         'epoch{0}\t{1} \n'.format(str(epoch + 1), str(train_epoch_loss_t1fapeaks)))

        # print("epoch:%d, train_loss_t1t2:%0.3f, train_dice_t1t2:%0.3f" % (epoch + 1, train_epoch_loss_t1t2, train_epoch_dice_t1t2))
        print("epoch:%d, train_loss_t1dec:%0.3f, train_dice_t1dec:%0.3f" % (epoch + 1, train_epoch_loss_fadecpeaks, train_epoch_dice_fadecpeaks))
        # print("epoch:%d, train_loss_t1fapeaks:%0.3f, train_dice_t1fapeaks:%0.3f" % (epoch + 1, train_epoch_loss_t1fapeaks, train_epoch_dice_t1fapeaks))

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
    # if 'outputs_FECC_t1t2' not in os.listdir(os.curdir):
    #     os.mkdir('outputs_FECC_t1t2')
    if 'outputs_FECC_t1dec' not in os.listdir(os.curdir):
        os.mkdir('outputs_FECC_t1dec')
    # if 'outputs5_FECC_t1fapeaks' not in os.listdir(os.curdir):
    #     os.mkdir('outputs5_FECC_t1fapeaks')

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
