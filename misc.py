import torch
import numpy as np
import os
from options import opt
from torch.autograd import Variable


def getLatestCheckpointName():
    if os.path.exists(opt.checkpoints_dir):
        print(opt.checkpoints_dir)
        file_name = os.listdir(opt.checkpoints_dir)
        file_names = []
        for i in file_name:
            if i == '.ipynb_checkpoints':
                continue
            file_names.append(i)
        print(f'file nmes is {file_names}')
        names_ext = [os.path.splitext(x) for x in file_names]
        checkpoint_names_G = []
        l = []
        for i in range(0, len(names_ext)):
            module = names_ext[i][1] == '.pt' and str(names_ext[i][0]).split('_')

            if module[0] == 'netG':
                checkpoint_names_G.append(int(module[1]))

        if len(checkpoint_names_G) == 0:
            return None

        g_index = max(checkpoint_names_G)
        ckp_g = None

        for i in file_names:
            if int(str(i).split('_')[1].split('.')[0]) == g_index and str(i).split('_')[0] == 'netG':
                ckp_g = i
                break

        return ckp_g