import argparse
import torch
import os

parser = argparse.ArgumentParser()
# print(parser)

parser.add_argument('--data_path', default='EUVP/1/')

parser.add_argument('--checkpoints_dir', default='./ckpts/')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_images', type=int, default=0)

parser.add_argument('--learning_rate_g', type=float, default=1e-3)
# parser.add_argument('--learning_rate_g', type=float, default=2e-04)
parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('--TRAIN.LR_SCHEDULER.NAME',default='cosine')
parser.add_argument('--TRAIN.WARMUP_EPOCHS',type=int,default=1) #swin transformer 是20
parser.add_argument('--MIN_LR',type=float,default=5e-6)
parser.add_argument('--WARMUP_LR',type=float,default=5e-7)


parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--end_epoch', type=int, default=1)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=512)

parser.add_argument('--beta1', type=float ,default=0.7)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.05)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_l1_loss', type=float, default=0.0)
parser.add_argument('--total_l1_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1)
parser.add_argument('--lambda_l1', type=float, default=0.5)
parser.add_argument('--lambda_vgg', type=float, default=0.02)

parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")
parser.add_argument('--testing_dir_inp', default='./EUVP/test samples/Inp/')
parser.add_argument('--testing_dir_gt', default='./EUVP/test samples/GTr/')

opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)