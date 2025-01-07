import os, random, shutil



def train_moveFile(fileDir_x_t1, fileDir_x_fa, fileDir_y):
    pathDir_x = os.listdir(fileDir_x_t1)  # 取图片的原始路径
    filenumber = len(pathDir_x)
    rate = 0.246  # 16938 x 0.2333 = 3951
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片

    sample = random.sample(pathDir_x, picknumber)  # 随机选取picknumber数量的样本图片

    ptahtarDir_x = os.listdir(tarDir_t1_x)
    tar_num = len(ptahtarDir_x)

    for name in sample:
        x_t1_name = name
        filename_num = name.replace('x_t1-data_', '')
        filename_num = filename_num.replace('.nii.gz', '')
        y_name = 'y-data_' + filename_num + '.nii.gz'
        x_fa_name = 'x_fa-data_' + filename_num + '.nii.gz'

        x_t1_tar_name = 'x_t1-data_' + str(tar_num) + '.nii.gz'
        x_fa_tar_name = 'x_fa-data_' + str(tar_num) + '.nii.gz'
        y_tar_name = 'y-data_' + str(tar_num) + '.nii.gz'

        shutil.copy(fileDir_x_t1 + x_t1_name, tarDir_t1_x + x_t1_tar_name)
        shutil.copy(fileDir_x_fa + x_fa_name, tarDir_fa_x + x_fa_tar_name)
        shutil.copy(fileDir_y + y_name, tarDir_y + y_tar_name)

        tar_num += 1
    return

if __name__ == '__main__':

    ### train
    # fileDir_t1_x = "./ON_mydata/train_data_nonlabel/x_t1_data/"    #源图片文件夹路径
    # fileDir_fa_x = "./ON_mydata/train_data_nonlabel/x_fa_data/"  # 源图片文件夹路径
    # fileDir_y    = "./ON_mydata/train_data_nonlabel/y_data/"  # 源图片文件夹路径
    #
    # tarDir_t1_x = "./ON_mydata/train_data/x_t1_data/"    #源图片文件夹路径
    # tarDir_fa_x = "./ON_mydata/train_data/x_fa_data/"  # 源图片文件夹路径
    # tarDir_y    = "./ON_mydata/train_data/y_data/"  # 源图片文件夹路径
    #
    # train_moveFile(fileDir_t1_x, fileDir_fa_x, fileDir_y)



    ### val
    fileDir_t1_x = "./ON_mydata/val_data_nonlabel/x_t1_data/"    #源图片文件夹路径
    fileDir_fa_x = "./ON_mydata/val_data_nonlabel/x_fa_data/"  # 源图片文件夹路径
    fileDir_y    = "./ON_mydata/val_data_nonlabel/y_data/"  # 源图片文件夹路径

    tarDir_t1_x = "./ON_mydata/val_data/x_t1_data/"    #源图片文件夹路径
    tarDir_fa_x = "./ON_mydata/val_data/x_fa_data/"  # 源图片文件夹路径
    tarDir_y    = "./ON_mydata/val_data/y_data/"  # 源图片文件夹路径

    train_moveFile(fileDir_t1_x, fileDir_fa_x, fileDir_y)

