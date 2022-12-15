import torch.nn as nn
import torch
import torch.nn.functional as F


class RSBranch(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(RSBranch, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1, bias=False),
        )

        self.up = nn.Sequential(
            Cutin(outchannels, outchannels),
            Cutin(outchannels, outchannels),
            Cutin(outchannels, outchannels),
            nn.Conv2d(outchannels, outchannels * 2 ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )

    def __call__(self, x):
        # print('-1', x.shape)
        x = self.down(x)
        # print('0', x.shape)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        # print('1', x.shape)
        x = self.up(x)
        # print('2', x.shape)

        return x


class Sobel(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Sobel, self).__init__()

        self.g_h = GradientConv(inchannels, outchannels, 'heng')
        self.g_s = GradientConv(inchannels, outchannels, 'shu')

    def __call__(self, x):
        g_h = self.g_h(x)
        g_s = self.g_s(x)

        out = abs(g_h) + abs(g_s)

        return out


class SobelBranch(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SobelBranch, self).__init__()

        # self.convin = nn.Conv2d(inchannels, outchannels, 3, 1, 1, bias=False)

        self.body = nn.Sequential(
            Sobel(inchannels, outchannels),
            nn.ReLU(),
            # nn.Conv2d(outchannels, outchannels, 3, 1, 1, bias=False),
            # nn.ReLU(),
        )

    def __call__(self, x):
        out = self.body(x)
        return out


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


class Cutin(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Cutin, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1, bias=False),
            nn.ReLU(),
        )

    def __call__(self, x):
        out = self.body(x)
        return out


class St_branch(nn.Module):
    def __init__(self, channels, size, stride):
        super(St_branch, self).__init__()
        # self.res = ResB(channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.MaxPool2d(kernel_size=size, padding=0),
            nn.ConvTranspose2d(channels, channels, 3, stride),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
        )

    def __call__(self, x):
        x1 = self.body1(x)

        diffY = x.size()[2] - x1.size()[2]
        diffX = x.size()[3] - x1.size()[3]

        x = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='constant', value=0)

        # x = self.res(x)

        return x


class GradientConv(nn.Module):
    def __init__(self, in_dim, out_dim, coretype):
        super(GradientConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        kernel = None
        if coretype is 'heng':
            kernel = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]
        elif coretype is 'shu':
            kernel = [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]

        kernel = torch.tensor(kernel).float().expand([out_dim, in_dim, 3, 3])
        # print('kernel_size:', kernel.shape)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        x1 = pad(x)
        x1 = F.conv2d(x1, self.weight)
        # print('x1.change:', x1.shape)

        return x1
