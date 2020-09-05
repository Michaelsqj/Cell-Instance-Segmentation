import torch
import os


def save_checkpoint(iteration, model, optimizer, lr_scheduler, output_path):
    state = {'iteration': iteration + 1,
             'state_dict': model.module.state_dict(),  # Saving torch.nn.DataParallel Models
             'optimizer': optimizer.state_dict(),
             'lr_scheduler': lr_scheduler.state_dict()}
    # Saves checkpoint to experiment directory
    filename = 'checkpoint_%04d.pth.tar' % (iteration + 1)
    filename = os.path.join(output_path, filename)
    torch.save(state, filename)


def update_checkpoint(checkpoint, model, optimizer, lr_scheduler, restart=False):
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

    if not restart:
        # update optimizer
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

        # update lr scheduler
        if 'lr_scheduler' in checkpoint.keys():
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # load iteration
        if 'iteration' in checkpoint.keys():
            start_iter = checkpoint['iteration']
            return start_iter
