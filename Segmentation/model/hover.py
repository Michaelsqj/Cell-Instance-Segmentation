"""
model same as the paper https://arxiv.org/abs/1812.06499
"""
import torch
from torch import nn
from torch.nn import functional as F


class _BNRelu(nn.Module):
    def __init__(self, num_features):
        super(_BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, inputs):
        return F.relu(self.bn(inputs), inplace=True)


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 expansion=4, preact=True):
        super(_ResidualUnit, self).__init__()
        self.preact = preact
        bottleneck_channels = out_channels // expansion

        self.bn_relu1 = _BNRelu(in_channels)

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, dilation=1, bias=False)
        self.bn_relu2 = _BNRelu(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               dilation=1, bias=False)
        self.bn_relu3 = _BNRelu(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, dilation=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=0, dilation=1,
                                      bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        out = self.bn_relu1(inputs) if self.preact else inputs
        shortcut = self.shortcut(inputs)
        out = self.bn_relu2(self.conv1(out))
        out = self.bn_relu3(self.conv2(out))
        out = self.conv3(out)
        out += shortcut

        return out


class _DenseUnit(nn.Module):
    def __init__(self, in_channels):
        super(_DenseUnit, self).__init__()
        self.bn_relu1 = _BNRelu(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.bn_relu2 = _BNRelu(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3,
                               padding=1, stride=1, dilation=1, bias=False)

    def forward(self, inputs):
        out = self.conv1(self.bn_relu1(inputs))
        out = self.conv2(self.bn_relu2(out))
        return torch.cat([out, inputs], dim=1)


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               padding=3, stride=1, dilation=1, bias=False)
        self.residual_block1 = nn.Sequential(
            _ResidualUnit(64, 256, stride=1, preact=False),
            _ResidualUnit(256, 256, stride=1, preact=True),
            _ResidualUnit(256, 256, stride=1, preact=True)
        )
        self.residual_block2 = nn.Sequential(
            _ResidualUnit(256, 512, stride=2, preact=False),
            _ResidualUnit(512, 512, stride=1, preact=True),
            _ResidualUnit(512, 512, stride=1, preact=True),
            _ResidualUnit(512, 512, stride=1, preact=True)
        )
        self.residual_block3 = nn.Sequential(
            _ResidualUnit(512, 1024, stride=2, preact=False),
            _ResidualUnit(1024, 1024, stride=1, preact=True),
            _ResidualUnit(1024, 1024, stride=1, preact=True),
            _ResidualUnit(1024, 1024, stride=1, preact=True),
            _ResidualUnit(1024, 1024, stride=1, preact=True),
            _ResidualUnit(1024, 1024, stride=1, preact=True)
        )
        self.residual_block4 = nn.Sequential(
            _ResidualUnit(1024, 2048, stride=2, preact=False),
            _ResidualUnit(2048, 2048, stride=1, preact=True),
            _ResidualUnit(2048, 2048, stride=1, preact=True)
        )
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x1 = self.residual_block1(x)
        x2 = self.residual_block2(x1)
        x3 = self.residual_block3(x2)
        x4 = self.residual_block4(x3)
        x4 = self.conv2(x4)

        return [x1, x2, x3, x4]


class _Decoder(nn.Module):
    def __init__(self, input_shape, in_channels):
        super(_Decoder, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=5,
                               padding=2, stride=1, dilation=1, bias=False)
        self.dense_block1 = nn.Sequential(
            _DenseUnit(256),
            _DenseUnit(256 + 32 * 1),
            _DenseUnit(256 + 32 * 2),
            _DenseUnit(256 + 32 * 3),
            _DenseUnit(256 + 32 * 4),
            _DenseUnit(256 + 32 * 5),
            _DenseUnit(256 + 32 * 6),
            _DenseUnit(256 + 32 * 7),
        )
        self.conv2 = nn.Conv2d(256 + 32 * 8, 512, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=5, padding=2,
                               bias=False)
        self.dense_block2 = nn.Sequential(
            _DenseUnit(128),
            _DenseUnit(128 + 32),
            _DenseUnit(128 + 32 * 2),
            _DenseUnit(128 + 32 * 3),
        )
        self.conv4 = nn.Conv2d(128 + 32 * 4, 256, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=5,
                               padding=2, stride=1, dilation=1, bias=False)

    # @torchsnooper.snoop()
    def forward(self, inputs):
        x = F.interpolate(inputs[-1], scale_factor=2)
        x = x + inputs[-2]
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = F.interpolate(self.conv2(x), scale_factor=2)
        x = self.conv3(x + inputs[-3])
        x = self.dense_block2(x)
        x = F.interpolate(self.conv4(x), scale_factor=2)
        x = self.conv5(x + inputs[-4])

        # x N*64*256*256
        return x


class _SegmentationHead(nn.Module):
    def __init__(self, head):
        super(_SegmentationHead, self).__init__()
        assert head in ['np', 'hv', 'nc']  # "Head must be 'np' or 'hv' or 'nc"
        self.head = head

        self.bn_relu = _BNRelu(num_features=64)
        if self.head in ['np', 'hv']:
            self.conv = nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1, dilation=1, bias=True)
        else:
            self.conv = nn.Conv2d(64, 5, kernel_size=1, padding=0, stride=1, dilation=1, bias=True)

    def forward(self, inputs):
        out = self.bn_relu(inputs)
        out = self.conv(out)
        return out


class HoverNet(nn.Module):
    def __init__(self):
        super(HoverNet, self).__init__()
        self.encoder = _Encoder()

        self.decoder_np = _Decoder((256, 256), 1024)
        self.decoder_hv = _Decoder((256, 256), 1024)
        self.decoder_nc = _Decoder((256, 256), 1024)

        self.head_np = _SegmentationHead(head='np')
        self.head_hv = _SegmentationHead(head='hv')
        self.head_nc = _SegmentationHead(head='nc')

    def forward(self, inputs):
        x = self.encoder(inputs)

        np = self.decoder_np(x)
        hv = self.decoder_hv(x)
        nc = self.decoder_nc(x)

        out_np = self.head_np(np)
        out_hv = self.head_hv(hv)
        out_nc = self.head_nc(nc)

        return out_np, out_hv, out_nc
