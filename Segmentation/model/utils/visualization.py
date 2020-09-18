import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

class visualizer():
    """
    tensorboard to visualize every loss, and image, output, mask
    """

    def __init__(self, out_path):
        self.writer = SummaryWriter(out_path + time.strftime("%m-%d-%H-%M", time.localtime()))

    def vis(self, iter, losses, image=None, output=None, mask=None):
        self.iter = iter
        self.scalar(losses)
        if image is not None:
            self.image(image, output, mask)

    def scalar(self, losses):
        self.writer.add_scalar('total_loss', sum(losses), global_step=self.iter)
        for i, loss in enumerate(losses):
            self.writer.add_scalar('loss_%d' % i, loss, global_step=self.iter)

    def image(self, images, outputs, masks):
        # images: Nx3xHxW, outputs: [NxCxHxW,...], masks: [NxCxHxW,...]
        for j in range(len(outputs)):
            for i in range(outputs[j].shape[1]):
                if type(outputs[j])==np.ndarray:
                    outputs[j]=torch.from_numpy(outputs[j])
                output = make_grid(outputs[j][:,i:i+1,...], padding=0, nrow=outputs[j].shape[0])
                self.writer.add_image('output%d_%d' % (j, i), output, global_step=self.iter)
        for j in range(len(masks)):
            for i in range(masks[j].shape[1]):
                if type(mask[j])==np.ndarray:
                    masks[j]=torch.from_numpy(masks[j])
                mask = make_grid(masks[j][:,i:i+1,...], padding=0, nrow=masks[j].shape[0])
                self.writer.add_image('mask%d_%d' % (j,i), mask, global_step=self.iter)
        if type(images)==np.ndarray:
            images=torch.from_numpy(images)
        image = make_grid(images, padding=0, nrow=images.shape[0])
        self.writer.add_image('image', image, global_step=self.iter)
        # todo figure out why the last sentence take effect at the next iteration