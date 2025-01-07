import numpy as np
import config_2d
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

batch_size = config_2d.BATCH_SIZE
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
n_classes = config_2d.NUM_CLASSES


def make_dirset(train_x_1_dir, train_x_2_dir, train_x_3_dir,train_x_4_dir,train_x_5_dir, train_y_dir):
    train_x_1_path = []
    train_x_2_path = []
    train_x_3_path = []
    train_x_4_path = []
    train_x_5_path = []
    train_y_path = []
    n = len(os.listdir(train_x_1_dir))
    for i in range(n):
        img_x_1 = os.path.join(train_x_1_dir, 'x_t1-data_%d.nii.gz' % i)
        train_x_1_path.append(img_x_1)
        img_x_2 = os.path.join(train_x_2_dir, 'x_t2-data_%d.nii.gz' % i)
        train_x_2_path.append(img_x_2)
        img_x_3 = os.path.join(train_x_3_dir, 'x_fa-data_%d.nii.gz' % i)
        train_x_3_path.append(img_x_3)
        img_x_4 = os.path.join(train_x_4_dir, 'x_dec-data_%d.nii.gz' % i)
        train_x_4_path.append(img_x_4)
        img_x_5 = os.path.join(train_x_5_dir, 'x_peaks-data_%d.nii.gz' % i)
        train_x_5_path.append(img_x_5)
        img_y = os.path.join(train_y_dir, 'y-data_%d.nii.gz' % i)
        train_y_path.append(img_y)
    return train_x_1_path, train_x_2_path, train_x_3_path,train_x_4_path,train_x_5_path, train_y_path


class MyTrainDataset(Dataset):
    def __init__(self, train_x_1_dir, train_x_2_dir, train_x_3_dir, train_x_4_path,train_x_5_path,train_y_dir, x_transform=None, z_transform=None,
                 y_transform=None):
        train_x_1_path, train_x_2_path, train_x_3_path,train_x_4_path,train_x_5_path, train_y_path = make_dirset(train_x_1_dir, train_x_2_dir,
                                                                                   train_x_3_dir,train_x_4_path,train_x_5_path, train_y_dir)
        self.train_x_1_path = train_x_1_path
        self.train_x_2_path = train_x_2_path
        self.train_x_3_path = train_x_3_path
        self.train_x_4_path = train_x_4_path
        self.train_x_5_path = train_x_5_path
        self.train_y_path = train_y_path

        self.x_transform = x_transform
        self.z_transform = z_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        x_1_path = self.train_x_1_path[index]
        x_2_path = self.train_x_2_path[index]
        x_3_path = self.train_x_3_path[index]
        x_4_path = self.train_x_4_path[index]
        x_5_path = self.train_x_5_path[index]
        y_path = self.train_y_path[index]

        img_x_1 = nib.load(x_1_path)
        img_x_1_data = img_x_1.get_fdata()
        x_1_are_Nans = np.isnan(img_x_1_data)
        img_x_1_data[x_1_are_Nans] = 0
        img_x_1_data = np.array(img_x_1_data, dtype='uint8')

        img_x_2 = nib.load(x_2_path)
        img_x_2_data = img_x_2.get_fdata()
        x_2_are_Nans = np.isnan(img_x_2_data)
        img_x_2_data[x_2_are_Nans] = 0
        img_x_2_data = np.array(img_x_2_data, dtype='uint8')

        img_x_3 = nib.load(x_3_path)
        img_x_3_data = img_x_3.get_fdata()
        x_3_are_Nans = np.isnan(img_x_3_data)
        img_x_3_data[x_3_are_Nans] = 0
        img_x_3_data = np.array(img_x_3_data, dtype='uint8')

        img_x_4 = nib.load(x_4_path)
        img_x_4_data = img_x_4.get_fdata()
        x_4_are_Nans = np.isnan(img_x_4_data)
        img_x_4_data[x_4_are_Nans] = 0
        img_x_4_data = np.array(img_x_4_data, dtype='float32')

        img_x_5 = nib.load(x_5_path)
        img_x_5_data = img_x_5.get_fdata()
        x_5_are_Nans = np.isnan(img_x_5_data)
        img_x_5_data[x_5_are_Nans] = 0
        img_x_5_data = np.array(img_x_5_data, dtype='float32')

        img_y = nib.load(y_path)
        img_y_data = img_y.get_fdata()

        if self.x_transform is not None:
            img_x_1_data = self.x_transform(img_x_1_data)
            img_x_2_data = self.x_transform(img_x_2_data)
            img_x_3_data = self.x_transform(img_x_3_data)

        if self.z_transform is not None:
            img_x_4_data = self.z_transform(img_x_4_data)
            img_x_5_data = self.z_transform(img_x_5_data)

        if self.y_transform is not None:
            img_y_data = self.y_transform(img_y_data)

        return img_x_1_data, img_x_2_data, img_x_3_data, img_x_4_data,img_x_5_data,img_y_data

    def __len__(self):  # 返回文件内train的nii数量
        return len(self.train_x_1_path)


# main
if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()
    ])
    z_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()

    ON_train_x1_dir = 'ON_mydata/train_data/x_t1_data/'
    ON_train_x2_dir = 'ON_mydata/train_data/x_t2_data/'
    ON_train_x3_dir = 'ON_mydata/train_data/x_fa_data/'
    ON_train_x4_dir = 'ON_mydata/train_data/x_dec_data/'
    ON_train_x5_dir = 'ON_mydata/train_data/x_peaks_data/'
    ON_train_y_dir = 'ON_mydata/train_data/y_data/'

    traindataset = MyTrainDataset(ON_train_x1_dir, ON_train_x2_dir, ON_train_x3_dir,ON_train_x4_dir,ON_train_x5_dir, ON_train_y_dir, x_transform=x_transforms,
                                  z_transform=z_transforms, y_transform=y_transforms)
    traindataloaders = DataLoader(traindataset, batch_size=4, shuffle=True, num_workers=0)

    for train_x1, train_x2, train_x3, train_x4,train_x5,train_y in traindataloaders:
        print('x1_shape=', train_x1.shape)
        print('x2_shape=', train_x2.shape)
        print('x3_shape=', train_x3.shape)
        print('x4_shape=', train_x4.shape)
        print('x5_shape=', train_x5.shape)
        print('label_shape=', train_y.shape)
        print("done")
    print("done")
