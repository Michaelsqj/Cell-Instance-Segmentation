import numpy as np
import imageio
import os
import torch


def read_img(path):
    suf = path.split('.')[-1]
    if suf == 'npy':
        image = np.load(path, allow_pickle=True)
    else:
        image = imageio.imread(path)
    return image


def save_checkpoint(model, optimizer, lr_scheduler, iteration, output_dir):
    state = {'iteration': iteration + 1,
             'state_dict': model.module.state_dict(),  # Saving torch.nn.DataParallel Models
             'optimizer': optimizer.state_dict(),
             'lr_scheduler': lr_scheduler.state_dict()}

    # Saves checkpoint to experiment directory
    filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
    filename = os.path.join(output_dir, filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr_scheduler):
    # load pre-trained model
    print('Load pretrained checkpoint: ', checkpoint)
    checkpoint = torch.load(checkpoint)
    print('checkpoints: ', checkpoint.keys())

    # update model weights
    if 'state_dict' in checkpoint.keys():
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.module.state_dict()  # nn.DataParallel
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.module.load_state_dict(model_dict)  # nn.DataParallel

    # update optimizer
    if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])

    # update lr scheduler
    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # load iteration
    if 'iteration' in checkpoint.keys():
        return checkpoint['iteration']
    else:
        return 0