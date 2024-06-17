import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from torchvision.transforms.functional import rotate,vflip
# from Deraining.online import data_degradation
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image

       0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = vflip(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = rotate(image,90)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = rotate(image,90)
        out = vflip(out)
    elif mode == 4:
        # rotate 180 degree
        out = rotate(image,180)
    elif mode == 5:
        # rotate 180 degree and flip
        out = rotate(image,180)
        out = vflip(out)
    elif mode == 6:
        # rotate 270 degree
        out = rotate(image,270)
    elif mode == 7:
        # rotate 270 degree and flip
        out = rotate(image,270)
        out = vflip(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        a = []
        b = []
        self.inp_filenames = []
        self.tar_filenames = []
        # setting the training dataset files here
        path = '/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/zhuanhaun/dataset/train/syn+real/input/'
        tar = '/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/zhuanhaun/dataset/train/syn+real/target'
        for x in os.listdir(path):
            filepath = os.path.join(path, x)
            tarpath = os.path.join(tar, x)
            a.append(filepath)
            b.append(tarpath)

        self.inp_filenames = a[:]
        self.tar_filenames = b[:]
        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target_real
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        tar_img = Image.open(tar_path)
        inp_img = Image.open(inp_path)
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        dd = random.randint(0, 7)
        inp_img1= data_augmentation(inp_img,dd)
        r, g, b = inp_img[0] + 1, inp_img[1] + 1, inp_img[2] + 1
        lr_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        lr_gray = torch.unsqueeze(lr_gray, 0)
        lr_white = 1 - lr_gray
        gray = torch.cat([inp_img, lr_gray], dim=0)
        white = torch.cat([inp_img, lr_white], dim=0)
        r1, g1, b1 = inp_img1[0] + 1, inp_img1[1] + 1, inp_img1[2] + 1
        lr_gray1 = 1. - (0.299 * r1 + 0.587 * g1 + 0.114 * b1) / 2.
        lr_gray1 = torch.unsqueeze(lr_gray1, 0)
        lr_white1 = 1 - lr_gray1
        gray1 = torch.cat([inp_img1, lr_gray1], dim=0)
        white1 = torch.cat([inp_img1, lr_white1], dim=0)
        return tar_img, inp_img, white, gray, inp_img1, white1,gray1, filename,lr_gray
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target_real

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        r, g, b = inp_img[0] + 1, inp_img[1] + 1, inp_img[2] + 1
        lr_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        lr_gray = torch.unsqueeze(lr_gray, 0)
        lr_white = 1 - lr_gray
        gray = torch.cat([inp_img, lr_gray], dim=0)
        white = torch.cat([inp_img, lr_white], dim=0)
        return tar_img, inp_img,white,gray, filename,lr_gray

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)


        return inp, filename
