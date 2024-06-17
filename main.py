import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from config import Config
# The opt file
opt = Config('/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/Low-light-rainy/training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
# The testing dataset files
img_path = '/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/zhuanhaun/dataset/test/input/'
targeet_path = '/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/zhuanhaun/dataset/test/target/'

img_list = sorted(os.listdir(img_path))
num_img = len(img_list)
import torch
torch.backends.cudnn.benchmark = True
import utils as utils
import os
from CR import ContrastLoss
from config import Config
from models.encoder2 import Convres
from restormer import Restormer_ours
pus = ','.join([str(i) for i in opt.GPU])
from torchvision.transforms import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import utils as utils
import time
import numpy as np
from model.common import VGGLoss
from data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
contrast_loss = torch.nn.CrossEntropyLoss().cuda()
# device = torch.device("cuda:0,1")
start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
device_ids = [i for i in range(torch.cuda.device_count())]
######### Model ###########
model_G1 = Convres()
model_G2 = Restormer_ours()

model_G1.cuda()
model_G2.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
new_lr = opt.OPTIM.LR_INITIAL
loss_vgg = VGGLoss().cuda()
optimizer_G1 = optim.Adam(model_G1.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
optimizer_G2 = optim.Adam(model_G2.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
######### Scheduler ###########
warmup_epochs = 90
scheduler_cosineG1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_G1, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                          eta_min=opt.OPTIM.LR_MIN)
scheduler_cosineG2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_G2, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                          eta_min=opt.OPTIM.LR_MIN)
scheduler_G1 = GradualWarmupScheduler(optimizer_G1, multiplier=1, total_epoch=warmup_epochs,
                                      after_scheduler=scheduler_cosineG1)
scheduler_G2 = GradualWarmupScheduler(optimizer_G2, multiplier=1, total_epoch=warmup_epochs,
                                      after_scheduler=scheduler_cosineG2)
scheduler_G1.step()
scheduler_G2.step()
######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = '/media/sr4/43cec1a8-a7e3-4f24-9dbb-b9b1b6950cf1/yjt/Low-light-rainy/checkpoint_our_res/model_290.pth'
    utils.load_checkpointG1(model_G1, path_chk_rest)
    utils.load_checkpointG2(model_G2, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    print("start_epoch=",start_epoch)
    utils.load_optimG1(optimizer_G1, path_chk_rest)
    utils.load_optimG2(optimizer_G2, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler_G1.step()
        scheduler_G2.step()
    new_lr = scheduler_G2.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_G1 = nn.DataParallel(model_G1, device_ids=device_ids)
    model_G2 = nn.DataParallel(model_G2, device_ids=device_ids)
######### Loss ###########
# criterion_char = Deraining.losses.CharbonnierLoss()
# criterion_edge = Deraining.losses.EdgeLoss()
ide_loss = torch.nn.L1Loss().cuda()
satu = ContrastLoss().cuda()
######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
transform = transforms.ToTensor()
print("The lr is:",scheduler_G1.get_lr()[0])
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    # The first stage, you should open the model_G1 and model_G2; the second stage, only open the model_G2
    # model_G1.eval()
    model_G1.train()
    model_G2.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        for param in model_G1.parameters():
            param.grad = None
        for param in model_G2.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        white = data[2].cuda()
        gray = data[3].cuda()
        input_ze = data[4].cuda()
        white1 = data[5].cuda()
        gray1 = data[6].cuda()

        ## The first stage: the model_G1 is training and the loss_con is the dual degradation loss in the paper.

        # mu1_1, dr1_1 = model_G1(input_ze, gray1, white1, flag='low')
        # mu2,dr2 = model_G1(target,gray,white, flag = 'clean')
        # 8, 384, 32, 32
        # da = ide_loss(mu1,mu1_1)
        # db = ide_loss(mu1,mu2)
        # loss_con = da/(db + 1e-7)

        ## The second stage: the model_G1 is fixed and the grad not backward.
        with torch.set_grad_enabled(False):
            mu1, dr1 = model_G1(input_, gray, white, flag='low')
        output = model_G2(input_,dr1)
        vgg = loss_vgg(output, target)
        ide = ide_loss(output, target)

        ## In the first stage, you should add the loss_con, e.g., DDLoss
        loss = ide + 0.1 * vgg

        loss.backward()
        # In the first stage open the optimizer_G1 and scheduler_G1
        # optimizer_G1.step()
        optimizer_G2.step()
        epoch_loss += loss.item()
    # scheduler_G1.step()
    scheduler_G2.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler_G2.get_lr()[0]))
    print("------------------------------------------------------------------")
    ## setting save frequency
    if epoch % 1 == 0 :
        torch.save({'epoch': epoch,
                    'state_dict_G1': model_G1.state_dict(),
                    'state_dict_G2': model_G2.state_dict(),
                    'optimizer_G1': optimizer_G1.state_dict(),
                    'optimizer_G2': optimizer_G2.state_dict()
                    }, os.path.join(model_dir,  'model_{}.pth'.format(epoch)))
        print("laileao")
        model_G1.eval()
        model_G2.eval()
        transform = transforms.ToTensor()
        PSNR = 0
        # testing stage
        for img in img_list:
            image = Image.open(img_path + '/' + img).convert('RGB')
            target = Image.open(targeet_path + '/' + img).convert('RGB')
            image = transform(image)
            target = transform(target)
            image = image.cuda()
            target = target.cuda()
            r, g, b = image[0] + 1, image[1] + 1, image[2] + 1
            lr_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            lr_gray = torch.unsqueeze(lr_gray, 0)
            lr_white = 1 - lr_gray
            [A, B, C] = image.shape
            image = image.reshape([1, A, B, C])
            [A, B, C] = target.shape
            target = target.reshape([1, A, B, C])
            [A, B, C] = lr_gray.shape
            lr_gray = lr_gray.reshape([1, A, B, C])
            [A, B, C] = lr_white.shape
            lr_white = lr_white.reshape([1, A, B, C])
            gray_test = torch.cat([image, lr_gray], dim=1)
            white_test = torch.cat([image, lr_white], dim=1)
            with torch.set_grad_enabled(False):
                mu, dr1 = model_G1(image,gray_test,white_test,'low')
                pre = model_G2(image, dr1)
            p_numpy = pre.squeeze(0).cpu().detach().numpy()
            label_numpy = target.squeeze(0).cpu().detach().numpy()
            psnr = peak_signal_noise_ratio(label_numpy, p_numpy, data_range=1)
            PSNR += psnr
        PSNR = PSNR / num_img
        print("PSNR =", PSNR)