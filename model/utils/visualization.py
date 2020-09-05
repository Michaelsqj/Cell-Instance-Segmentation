import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


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
        image = make_grid(images.squeeze(), padding=0, nrow=images.shape[0])
        self.writer.add_image('image', image, global_step=self.iter)
        for i in range(outputs.shape[1]):
            output = make_grid(outputs[i].squeeze(), padding=0, nrow=outputs.shape[0])
            self.writer.add_image('output%d' % i, output, global_step=self.iter)
        for i in range(outputs.shape[1]):
            mask = make_grid(masks[i].squeeze(), padding=0, nrow=masks.shape[0])
            self.writer.add_image('mask%d' % i, mask, global_step=self.iter)
