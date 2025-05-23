'''
prepare_data_path()	给定一个文件夹路径，收集所有图像文件的完整路径（支持 .bmp .tif .jpg .png）
imresize()	使用 PIL 对图像进行缩放（带有插值方式）
Fusion_dataset(Dataset)	PyTorch 的自定义数据集类，核心：加载图像 → 灰度处理 → 缩放 → 归一化 → 转 tensor
'''

import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
from numpy import asarray


## 返回完整的文件路径(data)以及最后一层的文件名(filenames)
def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)    # os.listdir()返回括号里文件夹下的所有文件及文件夹的名称列表
    data_dir = dataset_path                 
    data = glob.glob(os.path.join(data_dir, "*.bmp"))     ## data是dataset_path目录下以.bmp结尾的所有文件名的列表
    # os.path.join(path1,path2)会将path1和path2拼接起来，并根据操作系统添加适当的路径分隔符
    # glob.glob()用来查找匹配括号里的指定模式的文件，返回的是包含这些文件名的列表；支持*，？，[]三个通配符，*可代表0个或多个字符，？代表1个字符，[]用于匹配指定范围内的字符，例如 [0-9] 就是匹配任何单个数字。
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))    
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()   # 默认先按靠前的字符的顺序排序
    filenames.sort()
    return data, filenames

class Fusion_dataset(Dataset):
    # 在Python的类定义中，__init__函数（构造函数）里可以通过self.xxx = ...的方式动态地为实例添加属性
    # __init__()括号里的参数用来接收外部的初始化值
    def __init__(self, split, dti_path=None, t1_path=None, length=0):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        # assert用于调试，在代码中插入断言判断条件是否为真，不是真的话抛出错误，显示信息为条件后面的语句'split must be "train"|"val"|"test"'
        self.filepath_dti = []    
        self.filenames_dti = []
        self.filepath_t1 = []
        self.filenames_t1 = []
        self.length = length  # This place can be set up as much as you want to train
        if split == 'train':
            data_dir_t1 = "/data2/wangchangmiao/liuxiaoshuai/IXI_2d/"  # the path of your data
            data_dir_dti = "/data2/wangchangmiao/liuxiaoshuai/IXI_2d/"  # the path of your data
            dirs = [d for d in os.listdir(data_dir_dti) if not d.startswith('.')]    ## 以.开头的文件是隐藏的，通常是一些配置文件，比如.bashrc，不需要修改，所以过滤掉
            # str.startswith('str1')---检查字符串str是否以str1开头，返回True/False            
            dirs.sort()
            for dir0 in dirs:
                subdirs = [d for d in os.listdir(os.path.join(data_dir_dti, dir0)) if not d.startswith('.')]
                for dir1 in subdirs:
                    req_path = os.path.join(data_dir_dti, dir0, dir1, 'DTI')
                    for file in os.listdir(req_path):
                        if file.startswith('.'):
                            continue
                        filepath_dti_ = os.path.join(req_path, file)
                        self.filepath_dti.append(filepath_dti_)
                        self.filenames_dti.append(file)
                        filepath_t1_ = filepath_dti_.replace('DTI', 'T1')
                        # str.replace(old_substr,new_substr)用于将字符串str中的旧的子字符串替换为新的子字符串
                        self.filepath_t1.append(filepath_t1_)
                        self.filenames_t1.append(file)
            self.split = split
            # self.length = len(self.filepath_dti)  #if you want to train all data in the dataset
        elif split == 'test':
            data_dir_t1 = t1_path
            data_dir_dti = dti_path
            self.filepath_t1, self.filenames_t1 = prepare_data_path(data_dir_t1)
            self.filepath_dti, self.filenames_dti = prepare_data_path(data_dir_dti)
            self.split = split

    # 让自定义类的实例能像列表、字典一样通过 obj[index] 或 obj[key] 获取数据。
    ## 当dataset应用dataloader后，for dti,ti in dataloader时返回的是经过__getitem__函数后的结果
    def __getitem__(self, index):    ## 实例名[index]等同于实例名.__getitem__(index)
        if self.split == 'train':
            t1_path = self.filepath_t1[index]
            dti_path = self.filepath_dti[index]

            image_t1=np.load(t1_path)     # float32?
            image_dti=np.load(dti_path)

            image_t1 = np.expand_dims(image_t1, axis=0)    
            # np.expand_dims(数组a,axis)---在指定的axis上插入一个新维度，比如数组a(一维数组)的shape是(3,),np.expand_dims(a,axis=0)后shape变为(1,3)
            # shape(2,2)---np.expand(a,axis=1)---变为shape(2,1,2)

            image_dti = np.expand_dims(image_dti, axis=0)

            name = self.filenames_t1[index]
            return (
                torch.tensor(image_t1),
                torch.tensor(image_dti),
            )
        elif self.split == 'test':
            t1_path = self.filepath_t1[index]
            dti_path = self.filepath_dti[index]
            image_t1 = np.load(t1_path)
            if image_t1 is None:
                raise ValueError(f"Failed to load image at {t1_path}")

            image_dti = np.load(dti_path)
            if image_dti is None:
                raise ValueError(f"Failed to load image at {dti_path}")

            # image_t1 = np.asarray(Image.fromarray(image_t1), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            image_t1 = np.expand_dims(image_t1, axis=0)
            image_dti = np.expand_dims(image_dti, axis=0)

            name = self.filenames_t1[index]
            return (
                torch.tensor(image_t1),
                torch.tensor(image_dti),
            )

    def __len__(self):
        return self.length
