#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from glob import glob
from torch.autograd import Variable
from models.vmamba_Fusion_efficross import VSSM_Fusion
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger


from loss import Fusionloss

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')    # 程序运行时不显示警告

def parse_args():
    # argparse可以帮助开发者轻松地编写用户友好的命令行接口。该模块的核心功能是围绕argparse.ArgumentParser实例构建的，
    parse = argparse.ArgumentParser()
    return parse.parse_args()

## RGB2YCrCb和YCrCb2RGB是用来进行颜色空间的转换，RGB对应R,G,B三通道，YCrCb对应Y,Cr,Cb三通道
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(    # transpose(a,b)---交换a,b两个维度，从0维度开始
        1, 2).reshape(-1, 3)  # (nhw,c)    # -1自动推断改维度的大小
    ## reshape之前是(Batch_size,height,width,cannel),cannel=3时，-1处刚好是batch_size*height*width,不是3的会报错，无法整除算不出-1处的值
    R = im_flat[:, 0]    # Opencv中用cv2.imread()通常是BGR，Matplotlib的imshow是RGB，Pytorch也通常是RGB
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B    ## 计算Y,Cr,Cb通道
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)    # torch.unsqueeze(input, dim)---用于在指定的维度上插入一个大小为1的新维度,squeeze:挤压,unsqueeze:扩展
    # 原始张量： tensor([[1, 2, 3], [4, 5, 6]])
    # 原始张量形状： torch.Size([2, 3])
    # 在0维度上插入新维度后的张量： tensor([[[1, 2, 3], [4, 5, 6]]])
    # 在0维度上插入新维度后的张量形状： torch.Size([1, 2, 3])
    # 在1维度上插入新维度后的张量： tensor([[[1, 2, 3]], [[4, 5, 6]]])
    # 在1维度上插入新维度后的张量形状： torch.Size([2, 1, 3])
    # 在2维度上插入新维度后的张量： tensor([[[1], [2], [3]], [[4], [5], [6]]])
    # 在2维度上插入新维度后的张量形状： torch.Size([2, 3, 1])
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()    ## Y,Cr,Cb原本是[batch_size*h*w,1]沿dim=1拼接后变为[batch_size*h*w,3]
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(num=0, logger=None):
    lr_start = 0.0002
    modelpth = 'model_last'
    Method = 'my_cross'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('VSSM_Fusion')()
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train',length=400)    ## train_dataset是一个Fusion_dataset类的实例化对象，img_vis, img_ir = train_dataset[0]后img_vis和img_ir才是tensor类型的数据，因为[]调用了__getitem__函数
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(    ## for vis, ir in train_loader:         print(type(vis))  # <class 'torch.Tensor'>
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()

    epoch = 2
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.0001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir) in enumerate(train_loader):
            try:
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                # image_vis_ycrcb = image_vis[:,0:1:,:,:]
                image_ir = Variable(image_ir).cuda()
                fusion_image = fusionmodel(image_vis, image_ir)

            except TypeError as e:
                print(f"Caught TypeError: {e}")


            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            optimizer.zero_grad()


            # fusion loss
            loss_fusion,  loss_in, ssim_loss, loss_grad= criteria_fusion(
                image_vis=image_vis, image_ir=image_ir, generate_img=
                fusion_image, i=num, labels=None
            )



            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=ssim_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='VSSM_Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
    print("training Done!")
