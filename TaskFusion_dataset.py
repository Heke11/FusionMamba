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

def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = asarray(arr)                            # np.asarray()将各种类型的输入数据转换为numpy数据类型
    im = Image.fromarray(numpydata, mode=mode)          # 三维数组a,a[0,1,2]----第0个矩阵第1行第2列            
    # Image.fromarray()把numpy数组转换成灰度图像或RGB图像，mode='L'转换成灰度图像,mode='RGB'转换成彩色图像；返回PIL图像
    # 数组中数值范围均从0到255，对于灰度图像，对应二维数组，越接近0越黑，越接近255灰度越亮；对于彩色图像，对应三位数组，0到255就是代表像素的值。
    ts = type(size)
    # type() 适用于任何Python对象，用于获取对象的类型信息  print(type(a)) # 输出: <class 'list'> / 输出: <class 'dict'>
    # dtype 是NumPy数组的一个属性，用于返回数组中数据元素的类型。NumPy数组中的所有元素必须属于同一数据类型，因此可以使用dtype属性。  print(c.dtype) # 输出: int32 / 输出: float64
    ## size是单个数
    if np.issubdtype(ts, np.signedinteger):
        # numpy.issubdtype(arg1, arg2)---判断啊arg1是不是arg2的子类型，返回true或false;比如np.int32是np.integer的子类型，但是float32不是float64的子类型
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
        # np.array()创建数组对象，tuple()将括号里的可迭代对象(列表、集合等)转换为元组类型；元组与列表相似，但是元组被创建后不能修改，元组不可变。
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    ## size包含两个数
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}  ## 数组转换为图像时的插值方法
    imnew = im.resize(size, resample=func[interp])
    # 使用的Pillow库中的Image.resize(目标size,resample=插值方法或方法对应的序号(Pillow库内部定义的会直接映射))
    return np.array(imnew)

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
    def __init__(self, split, ir_path=None, vi_path=None, length=0):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        # assert用于调试，在代码中插入断言判断条件是否为真，不是真的话抛出错误，显示信息为条件后面的语句'split must be "train"|"val"|"test"'
        self.filepath_ir = []    
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length  # This place can be set up as much as you want to train
        if split == 'train':
            data_dir_vis = "/KAIST/"  # the path of your data
            data_dir_ir = "/KAIST/"  # the path of your data
            dirs = [d for d in os.listdir(data_dir_ir) if not d.startswith('.')]    ## 以.开头的文件是隐藏的，通常是一些配置文件，比如.bashrc，不需要修改，所以过滤掉
            # str.startswith('str1')---检查字符串str是否以str1开头，返回True/False            
            dirs.sort()
            for dir0 in dirs:
                subdirs = [d for d in os.listdir(os.path.join(data_dir_ir, dir0)) if not d.startswith('.')]
                for dir1 in subdirs:
                    req_path = os.path.join(data_dir_ir, dir0, dir1, 'lwir')
                    for file in os.listdir(req_path):
                        if file.startswith('.'):
                            continue
                        filepath_ir_ = os.path.join(req_path, file)
                        self.filepath_ir.append(filepath_ir_)
                        self.filenames_ir.append(file)
                        filepath_vis_ = filepath_ir_.replace('lwir', 'visible')
                        # str.replace(old_substr,new_substr)用于将字符串str中的旧的子字符串替换为新的子字符串
                        self.filepath_vis.append(filepath_vis_)
                        self.filenames_vis.append(file)
            self.split = split
            # self.length = len(self.filepath_ir)  #if you want to train all data in the dataset
        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split

    # 让自定义类的实例能像列表、字典一样通过 obj[index] 或 obj[key] 获取数据。
    def __getitem__(self, index):    ## 实例名[index]等同于实例名.__getitem__(index)
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = cv2.imread(vis_path)    # cv1.imread(filename,读取方式(cv2.IMREAD_COLOR或1--彩色图像/cv2.IMREAD_GRAYSCALE或0--灰度图像))，返回numpy数组，数组的shape:(高度，宽度，通道数)，size是图像的像素数，size = height × width × channels
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)    ## cvtColor进行颜色空间转换，BGR2GRAY将彩色转为灰色，3通道变1通道
            # if image_vis is None:
            #     raise ValueError(f"Failed to load image at {vis_path}")
            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)

            image_ir = cv2.imread(ir_path,0)
            if image_ir is None:
                raise ValueError(f"Failed to load image at {ir_path}")

            image_ir, image_vis = self.resize(image_ir, image_vis, [256, 256], [256, 256])  


            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)    
            # np.expand_dims(数组a,axis)---在指定的axis上插入一个新维度，比如数组a(一维数组)的shape是(3,),np.expand_dims(a,axis=0)后shape变为(1,3)
            # shape(2,2)---np.expand(a,axis=1)---变为shape(2,1,2)

            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
            )
        elif self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = cv2.imread(vis_path)
            gray_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)
            if image_vis is None:
                raise ValueError(f"Failed to load image at {vis_path}")

            image_ir = cv2.imread(ir_path, 0)
            if image_ir is None:
                raise ValueError(f"Failed to load image at {ir_path}")

            # image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)
            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
            )

    def __len__(self):
        return self.length

    def resize(self, data, data2, crop_size_img, crop_size_label):
        data = imresize(data, crop_size_img, interp='bicubic')
        data2 = imresize(data2, crop_size_label, interp='bicubic')
        return data, data2
