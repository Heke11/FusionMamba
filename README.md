# FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba


[Arxiv](https://arxiv.org/abs/2404.09498)| [Code](https://github.com/millieXie/FusionMamba) | 

## 1. Create Environment

conda create -n FusionMamba python=3.8

conda activate FusionMamba 

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 

pip install packaging pip install timm==0.4.12 

pip install pytest chardet yacs termcolor

pip install submitit tensorboardX 

pip install triton==2.0.0

pip install causal_conv1d==1.0.0 # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install mamba_ssm==1.0.1 # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

## 2. Prepare Your Dataset
dataset
```

/dataset/
        set00-setXX/
                        V000-VXXX/
                                        IRimages
                                        VISimages

```

## 3. Pretrain Weights

This file is an infrared and visible light model file. Its training hyperparameters have been written so you can use them as you wish. 
link：https://pan.baidu.com/s/1wHqLA3R2ovZyEfTC00wwsg?pwd=6yr2 
password：6yr2
 
 ## 4.Train
 
```
python train.py
```
## 5.Test

```
python test.py
```

## 6.Citation

@article{xie2024fusionmamba,
  title={Fusionmamba: Dynamic feature enhancement for multimodal image fusion with mamba},
  author={Xie, Xinyu and Cui, Yawen and Ieong, Chio-In and Tan, Tao and Zhang, Xiaozhi and Zheng, Xubin and Yu, Zitong},
  journal={arXiv preprint arXiv:2404.09498},
  year={2024}
}
