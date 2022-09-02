import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from options import opt

"""# Channel and Spatial Attention"""

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = 1
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_c
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape # torch.Size([1, 256, 256, 128])
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class selfAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(selfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x): # torch.Size([1024, 64, 128])
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x): # torch.Size([1024, 64, 36])
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        x = x.reshape(3,B,N,C//3) # torch.Size([3, 1024, 64, 12])
        Red = torch.unsqueeze(x[0, :, :,:], dim=0) # torch.Size([1, 1024, 64, 12])
        Green = torch.unsqueeze(x[ 1,:, :, :], dim=0)
        Blue = torch.unsqueeze(x[2, :, :, :], dim=0)
        Red = torch.cat((Red,Red,Red))
        Red = Red.reshape(B,N,C)
        Green = torch.cat((Green, Green, Green))
        Green = Green.reshape(B, N, C)
        Blue = torch.cat((Blue, Blue, Blue))
        Blue = Blue.reshape(B, N, C)
        qkv_1 = self.qkv(Red).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v =  qkv_1[1], qkv_1[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_2 = self.qkv(Green).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_3 = self.qkv(Blue).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = torch.add(qkv_2[0],qkv_3[0])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # torch.Size([1024, 2, 64, 64])
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SELFATT_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SELFATT_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.attn = selfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, x_size): # torch.Size([1, 65536, 32])  ## x:[Batch,h*w,channel] x_size [H,W]
        H, W = x_size
        B, L, C = x.shape
        x = self.norm1(x)
        x = x.view(B, H, W, C)  ## [B,H,W,C]

        #### num_windows*B, window_size, window_size, C
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C # torch.Size([1024, 8, 8, 32])
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x = self.drop_path(self.mlp(self.norm2(self.drop_path(self.attn(x_windows))))) # attn_block:torch.Size([1024, 64, 32])
        x = x.view(B, H * W, C)
        return x

class ATT_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(ATT_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x,x_size):
        H, W = x_size
        B, L, C = x.shape
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        ###  num_windows*B, window_size, window_size, C   :: window size ::[8*8]
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C # torch.Size([1024, 8, 8, 32])
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x = self.drop_path(self.mlp(self.norm2(self.drop_path(self.attn(x_windows))))) # attn_block:torch.Size([1024, 64, 32])
        x = x.view(B, H * W, C)
        return x

class self_pxp(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, patch_size):
        super(self_pxp, self).__init__()

        print('self-attention!!')
        img_size = (256, 256)
        self.conv_first = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)
        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_c=out_ch, embed_dim=out_ch,
            norm_layer=norm_layer)

        self.layer = SELFATT_Block(dim=out_ch,
                               num_heads=1, window_size=8,
                               qkv_bias=True, qk_scale=None, attn_drop_ratio=0.)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=out_ch, embed_dim=out_ch,
            norm_layer=norm_layer)

    def forward(self, input):
        img_size = (256, 256)
        flops = self.relu(self.bn(self.conv_first(input))) # torch.Size([1, 32, 256, 256])
        p_em = self.patch_embed(flops)
        return self.patch_unembed(self.layer(p_em, img_size), img_size)

class attn_pxp(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, patch_size):
        super(attn_pxp, self).__init__()

        print('self-attention!!')
        img_size = (256, 256)
        self.conv_first = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)
        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_c=out_ch, embed_dim=out_ch,
            norm_layer=norm_layer)

        self.layer = ATT_Block(dim=out_ch,
                               num_heads=1, window_size=8,
                               qkv_bias=True, qk_scale=None, attn_drop_ratio=0.)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=out_ch, embed_dim=out_ch,
            norm_layer=norm_layer)

    def forward(self, input):
        img_size = (256, 256)
        flops = self.relu(self.bn(self.conv_first(input)))
        p_em = self.patch_embed(flops)
        return self.patch_unembed(self.layer(p_em, img_size), img_size)

class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()

        print("Color correction module for underwater images")

        self.layer1_1 = Conv2D_pxp(1, 32, 3, 1, 1)

        self.layer2_1 = self_pxp(32, 32, 3, 1, 1, 6)
        self.layer2_2 = attn_pxp(96, 96, 3, 1, 1, 6)

        self.layer3_1 = self_pxp(32, 96, 3, 1, 1, 6)
        self.layer3_2 = attn_pxp(160, 36, 3, 1, 1, 6)

        self.layer4_1 = Conv2D_pxp(228, 1, 1, 1, 0)

        self.d_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.PReLU(32)

        self.global_attn_rgb = CBAM(35)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=35, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=3)
        self.d_relu2 = nn.PReLU(3)

    def forward(self, input): # torch.Size([1, 3, 256, 256])

        input_1 = torch.unsqueeze(input[:, 0, :, :], dim=1) #torch.Size([1, 1, 256, 256])
        input_2 = torch.unsqueeze(input[:, 1, :, :], dim=1)
        input_3 = torch.unsqueeze(input[:, 2, :, :], dim=1)

        # layer 1
        l1_1 = self.layer1_1(input_1)  # 1,32,160,120
        l1_2 = self.layer1_1(input_2)
        l1_3 = self.layer1_1(input_3)

        inputl2 = torch.cat((l1_1, l1_2), 1)
        inputl2 = torch.cat((inputl2, l1_3), 1)

        l2_1 = self.layer2_2(inputl2)
        l2_2 = self.layer2_1(l1_2)
        l2_3 = self.layer2_1(l1_3)

        inputl3 = torch.cat((l2_1, l2_2), 1)
        inputl3 = torch.cat((inputl3, l2_3), 1)

        # Input to layer 3
        l3_1 = self.layer3_2(inputl3)
        l3_2 = self.layer3_1(l2_2)
        l3_3 = self.layer3_1(l2_3)

        input_l4 = torch.cat((l3_1, l3_2), 1)
        input_l4 = torch.cat((input_l4, l3_3), 1)

        l4_1 = self.layer4_1(input_l4)
        l4_2 = self.layer4_1(input_l4)
        l4_3 = self.layer4_1(input_l4)

        temp_d1 = torch.add(input_1, l4_1)
        temp_d2 = torch.add(input_2, l4_2)
        temp_d3 = torch.add(input_3, l4_3)

        input_d1 = torch.cat((temp_d1, temp_d2), 1)
        input_d1 = torch.cat((input_d1, temp_d3), 1)
        # print(input_d1.shape)

        # decoder
        output_d1 = self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))
        output_d1 = self.global_attn_rgb(torch.cat((output_d1, input_d1), 1))
        final_output = self.d_relu2(self.d_bn2(self.d_conv2(output_d1)))
        # print(f'after:{final_output.shape}')
        return final_output