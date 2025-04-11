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

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, length=0):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length  # This place can be set up as much as you want to train
        if split == 'train':
            data_dir_vis = "/KAIST/"  # the path of your data
            data_dir_ir = "/KAIST/"  # the path of your data
            dirs = [d for d in os.listdir(data_dir_ir) if not d.startswith('.')]
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

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = cv2.imread(vis_path)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)
            # if image_vis is None:
            #     raise ValueError(f"Failed to load image at {vis_path}")
            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)

            image_ir = cv2.imread(ir_path,0)
            if image_ir is None:
                raise ValueError(f"Failed to load image at {ir_path}")

            image_ir, image_vis = self.resize(image_ir, image_vis, [256, 256], [256, 256])  


            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)

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
