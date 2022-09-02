import dataset as dataset
from vgg import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from options import opt, device
from models import *
from misc import *
import re
import sys
from test import test
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from measure_ssim_psnr import SSIMs_PSNRs
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.tensorboard import SummaryWriter
## local libs
from measure_uiqm import measure_UIQMs
from timm.scheduler.cosine_lr import CosineLRScheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':

    netG = CC_Module()
    netG.to(device)

    mse_loss = nn.MSELoss()
    L1 = nn.L1Loss()

    vgg = Vgg16(requires_grad=False).to(device)

    optim_g = optim.AdamW(netG.parameters(),
                          lr=opt.learning_rate_g,
                          betas=(opt.beta1, opt.beta2),
                          weight_decay=opt.wd_g, eps=opt.eps)

    lr_scheduler = CosineLRScheduler(
        optimizer=optim_g,
        t_initial=int(opt.end_epoch),

        lr_min=opt.MIN_LR,  # 5e-6
        warmup_lr_init=opt.WARMUP_LR,  # 5e-7
        warmup_t=0,  # 0
        cycle_limit=1,
        t_in_epochs=False,
    )

    dataset = dataset.Dataset_Load(data_path=opt.data_path,
                                   transform=dataset.ToTensor()
                                   )

    batches = int(dataset.len / opt.batch_size)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    models_loaded = getLatestCheckpointName()
    latest_checkpoint_G = models_loaded

    print('loading model for generator ', latest_checkpoint_G)
    # latest_checkpoint_G = None

    if latest_checkpoint_G == None:
        start_epoch = 1
        print('No checkpoints found for netG and netD! retraining')

    else:
        checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))
        # model.load_state_dict(torch.load())
        start_epoch = checkpoint_g['epoch'] + 1
        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        print(f'start:{start_epoch}')

        print('Restoring model from checkpoint ' + str(start_epoch))

    netG.train()
    from tensorboardX import SummaryWriter  # tensorboard

    writer = SummaryWriter('./result/')

    for epoch in range(start_epoch, opt.end_epoch + 1):

        opt.total_mse_loss = 0.0
        opt.total_vgg_loss = 0.0
        opt.total_G_loss = 0.0

        torch.manual_seed(2021)
        for i_batch, sample_batched in enumerate(dataloader):
            hazy_batch = sample_batched['hazy']  # trainA
            clean_batch = sample_batched['clean']  # trainB  # torch.size([1,3,256,256])

            hazy_batch = hazy_batch.to(device)  # images = images.cuda(non_blocking=True)
            clean_batch = clean_batch.to(device)  # torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            optim_g.zero_grad()

            pred_batch = netG(hazy_batch)  # outputs = nn.Conv2d(imgaes)

            batch_mse_loss = torch.mul(opt.lambda_mse, mse_loss(pred_batch, clean_batch))
            batch_mse_loss.backward(retain_graph=True)

            clean_vgg_feats = vgg(normalize_batch(clean_batch))  # loss = criterion(outputs, target)
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg_loss = torch.mul(opt.lambda_vgg, mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2))
            batch_vgg_loss.backward()

            opt.batch_mse_loss = batch_mse_loss.item()
            opt.total_mse_loss += opt.batch_mse_loss

            opt.batch_vgg_loss = batch_vgg_loss.item()
            opt.total_vgg_loss += opt.batch_vgg_loss

            opt.batch_G_loss = opt.batch_mse_loss + opt.batch_vgg_loss
            opt.total_G_loss += opt.batch_G_loss

            optim_g.step()

            lr_scheduler.step(epoch)

            print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch + 1) + '/' + str(batches) + ') | g_mse: ' + str(
                opt.batch_mse_loss) + ' | g_vgg: ' + str(opt.batch_vgg_loss),
                  end='', flush=True)
        writer.add_scalar('mse', opt.total_mse_loss, epoch)  # tensorboard
        writer.add_scalar('vgg', opt.total_vgg_loss, epoch)
        writer.add_scalar('total', opt.total_G_loss, epoch)

        print('\nFinished ep. %d, lr = %.6f, total_mse = %.6f,total_vgg = %.6f,total_loss = %.6f' % (
            epoch, get_lr(optim_g), opt.total_mse_loss, opt.total_vgg_loss, opt.total_G_loss))

        torch.save({'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optim_g.state_dict(),
                    'mse_loss': opt.total_mse_loss,
                    'vgg_loss': opt.total_vgg_loss,
                    'opt': opt,
                    'total_loss': opt.total_G_loss}, os.path.join(opt.checkpoints_dir, 'netG_' + str(epoch) + '.pt'))
        test(netG, epoch)

        SSIM_measures, PSNR_measures, rmse_measures = SSIMs_PSNRs("./EUVP/test samples/GTr/",
                                                                  './facades/' + 'netG_' + str(epoch) + '/')
        print("SSIM on {0} samples".format(len(SSIM_measures)))
        print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
        writer.add_scalar('SSIM_mean', np.mean(SSIM_measures), epoch)
        writer.add_scalar('SSIM_std', np.std(SSIM_measures), epoch)
        print("PSNR on {0} samples".format(len(PSNR_measures)))
        print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
        writer.add_scalar('PSNR_mean', np.mean(PSNR_measures), epoch)
        writer.add_scalar('PSNR_std', np.std(PSNR_measures), epoch)

        # UIQMs of the enhanceded output images
        gen_dir = './facades/' + 'netG_' + str(epoch) + '/'
        gen_uqims = measure_UIQMs(gen_dir)
        print("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
        writer.add_scalar('gen_mean', np.mean(gen_uqims), epoch)
        writer.add_scalar('gen_std', np.std(gen_uqims), epoch)
