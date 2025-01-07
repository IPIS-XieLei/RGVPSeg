import torch
import nibabel as nib
import numpy

train_t1 = "/home/AVP Seg/2D_FusionNet/ON_mydata/train_data/x_t1_data/x_t1-data_0.nii.gz"
train_fa = "/home/AVP Seg/2D_FusionNet/ON_mydata/train_data/x_fa_data/x_fa-data_0.nii.gz"
train_peaks = "/home/AVP Seg/2D_FusionNet/ON_mydata/train_data/x_peaks_data/x_peaks-data_0.nii.gz"
val_t1 = "/home/AVP Seg/2D_FusionNet/ON_mydata/val_data/x_t1_data/x_t1-data_0.nii.gz"
val_fa = "/home/AVP Seg/2D_FusionNet/ON_mydata/val_data/x_fa_data/x_fa-data_0.nii.gz"
val_peaks = "/home/AVP Seg/2D_FusionNet/ON_mydata/val_data/x_peaks_data/x_peaks-data_0.nii.gz"
test_t1 = "/home/AVP Seg/2D_FusionNet/ON_mydata/test_data/test_106016/x_t1_data/x_t1-data_0.nii.gz"
test_fa = "/home/AVP Seg/2D_FusionNet/ON_mydata/test_data/test_106016/x_fa_data/x_fa-data_0.nii.gz"
test_peaks = "/home/AVP Seg/2D_FusionNet/ON_mydata/test_data/test_106016/x_peaks_data/x_peaks-data_0.nii.gz"

train_t1_data = nib.load(train_t1)
train_t1_data = train_t1_data.get_fdata()
train_fa_data = nib.load(train_fa)
train_fa_data = train_fa_data.get_fdata()
train_peaks_data = nib.load(train_peaks)
train_peaks_data = train_peaks_data.get_fdata()
print(type(train_t1_data))
print(type(train_fa_data))
print(type(train_peaks_data))
val_t1_data = nib.load(val_t1)
val_t1_data = val_t1_data.get_fdata()
val_fa_data = nib.load(val_fa)
val_fa_data = val_fa_data.get_fdata()
val_peaks_data = nib.load(val_peaks)
val_peaks_data = val_peaks_data.get_fdata()
print(type(val_t1_data))
print(type(val_fa_data))
print(type(val_peaks_data))
test_t1_data = nib.load(test_t1)
test_t1_data = test_t1_data.get_fdata()
test_fa_data = nib.load(test_fa)
test_fa_data = test_fa_data.get_fdata()
test_peaks_data = nib.load(test_peaks)
test_peaks_data = test_peaks_data.get_fdata()
print(type(test_t1_data))
print(type(test_fa_data))
print(type(test_peaks_data))
