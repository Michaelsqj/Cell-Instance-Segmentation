import torch.nn as nn
from Segmentation.model.module import _Decoder, _Encoder, Scale_attention


class HoverNet(nn.Module):
    def __init__(self):
        super(HoverNet, self).__init__()
        self.encoder = _Encoder()

        self.decoder_np = _Decoder((256, 256), 1024)
        self.decoder_hv = _Decoder((256, 256), 1024)
        self.decoder_nc = _Decoder((256, 256), 1024)

        self.scale_np = Scale_attention(512, 256, 64, 2)
        self.scale_hv = Scale_attention(512, 256, 64, 2)
        self.scale_nc = Scale_attention(512, 256, 64, 5)

    def forward(self, inputs):
        x1, x2, x3, x4 = self.encoder(inputs)

        np_x1, np_x2, np_x3 = self.decoder_np(x1, x2, x3, x4)
        hv_x1, hv_x2, hv_x3 = self.decoder_hv(x1, x2, x3, x4)
        nc_x1, nc_x2, nc_x3 = self.decoder_nc(x1, x2, x3, x4)

        out_np = self.scale_np(np_x1, np_x2, np_x3)
        out_hv = self.scale_hv(hv_x1, hv_x2, hv_x3)
        out_nc = self.scale_nc(nc_x1, nc_x2, nc_x3)
        return out_np, out_hv, out_nc
