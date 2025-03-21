import torch
import torch.nn as nn
# from basicsr.models.archs import recons_video81 as recons_video
# from basicsr.models.archs import flow_pwc82 as flow_pwc
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.common import make_layer
from mmcv.runner import load_checkpoint

def make_model(opt):
    device = 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = opt['pretrain_models_dir'] + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return GShiftNet()
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.relu = nn.PReLU() # nn.ReLU(inplace=True)
        # if res_scale == 1.0:
        #     self.init_weights()
        self.CA = CALayer(mid_channels, 4, bias=False)

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        identity = x
        out = self.CA(self.conv2(self.relu(self.conv1(x))))
        return identity + out * self.res_scale

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        # main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(nn.PReLU())
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
    
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
class RepConv(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv, self).__init__()
        self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat//8)
    def forward(self, x):
        res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_1 + res_2 + x
class RepConv2(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv2, self).__init__()
        #self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)
    def forward(self, x):
        #res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_2 + x
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)
class CAB1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB1, self).__init__()
        modules_body = []
        scale_factor = 1
        n_scale_feat = int(scale_factor * n_feat)
        self.norm = LayerNorm2d(n_feat)
        modules_body.append(conv(n_feat, n_scale_feat*2, 1, bias=bias))
        # modules_body.append(nn.GELU())
        # modules_body.append(nn.PReLU())
        modules_body.append(RepConv2(n_scale_feat*2, kernel_size, bias))
        # modules_body.append(nn.Conv2d(n_scale_feat*2, n_scale_feat*2, 3, bias=bias, padding=1, groups=n_scale_feat*2))
        modules_body.append(SimpleGate())
        # modules_body.append(CALayer2(n_scale_feat, reduction, bias=bias))
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2*n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))

        # self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = self.body(self.norm(x))
        # res = self.CA(res)
        res = x + res * self.beta
        return res
class CAB2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, add_channel=0):
        super(CAB2, self).__init__()
        modules_body = []
        scale_factor = 1
        self.n_feat = n_feat
        self.add_channel = add_channel
        n_scale_feat = int(scale_factor * n_feat)
        # self.conv1 = conv(n_feat, n_scale_feat//2, 1, bias=bias)
        # self.conv2 = conv(n_feat//2+self.add_channel, n_scale_feat//2, 1, bias=bias)
        # self.conv1 = nn.Sequential(conv(self.add_channel, self.add_channel, 1, bias=bias), act)
        self.conv1 = nn.Conv2d(self.add_channel, self.add_channel, 3, bias=bias, padding=1, groups=self.add_channel)
        self.norm = LayerNorm2d(self.add_channel + n_feat)
        modules_body.append(conv(n_feat + self.add_channel, n_scale_feat*2, 1, bias=bias))
        # modules_body.append(nn.GELU())
        # modules_body.append(nn.PReLU())
        modules_body.append(RepConv2(n_scale_feat*2, kernel_size, bias))
        # modules_body.append(nn.Conv2d(n_scale_feat*2, n_scale_feat*2, 3, bias=bias, padding=1, groups=n_scale_feat*2))
        modules_body.append(SimpleGate())
        # modules_body.append(CALayer2(n_scale_feat, reduction, bias=bias))
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2*n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))

        # self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)
        # self.non_linear = nn.GELU()
        # body2 = []
        # body2.append(conv(n_feat, n_feat, 1, bias=bias))
        # body2.append(nn.GELU())
        # body2.append(conv(n_feat, n_feat, 1, bias=bias))
        # self.body2 = nn.Sequential(*body2)


    def forward(self, x_input):
        shortcut, hw = x_input[:,0:self.n_feat], x_input[:,self.n_feat:]
        hw = self.conv1(hw)
        res = self.body(self.norm(torch.cat((shortcut, hw), dim=1)))
        # res = self.CA(res)
        res = shortcut + res * self.beta
        return res
class PixelShufflePack(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        # self.init_weights()

    def init_weights(self):
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x
## Original Resolution Block (ORB)
class CABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABs, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, 3, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)
# RDB-based RNN cell
class shallow_cell(nn.Module):
    def __init__(self, n_features):
        super(shallow_cell, self).__init__()
        self.n_feats = n_features
        act = nn.PReLU()
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(3, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
        #                           nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
        # self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True),
        #                    nn.PReLU())
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        x = self.down(x)
        return x
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # U-net skip
        # self.skip_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        # self.skip_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        # self.skip_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
        #                            bias=bias)

        # self.skip_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        # self.skip_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        # self.skip_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
        #                            bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]
def generate_kernels(h=11,l=80,n=10):
    kernels = torch.zeros(l,1,h,h).to(torch.device('cuda'))
    n2 = 2
    n1 = n-2*n2
    kernels[0*n2:1*n2,:,0,0] = 1
    kernels[1*n2:2*n2,:,0,h//4] = 1
    kernels[2*n2:3*n2,:,0,h//2] = 1
    kernels[3*n2:4*n2,:,0,3*h//4] = 1
    kernels[4*n2:5*n2,:,0,h-1] = 1
    kernels[5*n2:6*n2,:,h-1,0] = 1
    kernels[6*n2:7*n2,:,h-1,h//4] = 1
    kernels[7*n2:8*n2,:,h-1,h//2] = 1
    kernels[8*n2:9*n2,:,h-1,3*h//4] = 1
    kernels[9*n2:10*n2,:,h-1,h-1] = 1
    kernels[10*n2:11*n2,:,h//4,0] = 1
    kernels[11*n2:12*n2,:,h//4,h-1] = 1
    kernels[12*n2:13*n2,:,h//2,0] = 1
    kernels[13*n2:14*n2,:,h//2,h-1] = 1
    kernels[14*n2:15*n2,:,3*h//4,0] = 1
    kernels[15*n2:16*n2,:,3*h//4,h-1] = 1
    kernels[16*n2+0*n1:16*n2+1*n1,:,h//4,h//4] = 1
    kernels[16*n2+1*n1:16*n2+2*n1,:,h//4,h//2] = 1
    kernels[16*n2+2*n1:16*n2+3*n1,:,h//4,3*h//4] = 1 
    kernels[16*n2+3*n1:16*n2+4*n1,:,h//2,h//4] = 1
    kernels[16*n2+4*n1:16*n2+5*n1,:,h//2,3*h//4] = 1
    kernels[16*n2+5*n1:16*n2+6*n1,:,3*h//4,h//4] = 1
    kernels[16*n2+6*n1:16*n2+7*n1,:,3*h//4,h//2] = 1
    kernels[16*n2+7*n1:16*n2+8*n1,:,3*h//4,3*h//4] = 1
    return kernels
class Encoder_shift_block(nn.Module):
    def __init__(self, n_features, kernel_size, reduction, bias=False, scale_unetfeats=48):
        super(Encoder_shift_block, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        number = n_feat // 2 // 8
        self.number = number
        self.encoder_level1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_2 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_3 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_4 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_5 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_6 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_7 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level1_1 = nn.Sequential(*self.encoder_level1_1)
        self.encoder_level1_2 = nn.Sequential(*self.encoder_level1_2)
        self.encoder_level1_3 = nn.Sequential(*self.encoder_level1_3)
        self.encoder_level1_4 = nn.Sequential(*self.encoder_level1_4)
        self.encoder_level1_5 = nn.Sequential(*self.encoder_level1_5)
        self.encoder_level1_6 = nn.Sequential(*self.encoder_level1_6)
        self.encoder_level1_7 = nn.Sequential(*self.encoder_level1_7)
        #self.shift_conv = nn.Conv2d(8*self.number,8*self.number,5, bias=False,padding=2,groups=8*self.number)
        #self.shift_conv.weight = torch.nn.Parameter(generate_kernels(h=5,l=8*self.number,n=self.number))
        #self.shift_conv.requires_grad = False
        # self.shift_conv1 = nn.Conv2d(8*self.number,8*self.number,21, bias=False,padding=10,groups=8*self.number)
        # self.shift_conv1.weight = torch.nn.Parameter(generate_kernels(h=21,l=8*self.number,n=self.number))
        # self.shift_conv1.requires_grad = False
    def spatial_shift2(self, hw):
        n2 = (self.number-1)//2
        n1 = self.number-2*n2
        s = 4
        s_list = []
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        s_out[:,0*n2:1*n2,s*2:,s*2:] = hw[:,0*n2:1*n2,:-s*2,:-s*2]
        s_out[:,1*n2:2*n2,s*2:,s:] = hw[:,1*n2:2*n2,:-s*2,:-s]
        s_out[:,2*n2:3*n2,s*2:,0:] = hw[:,2*n2:3*n2,:-s*2,:]
        s_out[:,3*n2:4*n2,s*2:,0:-s] = hw[:,3*n2:4*n2,:-s*2,s:]
        s_out[:,4*n2:5*n2,s*2:,0:-s*2] = hw[:,4*n2:5*n2,:-s*2,s*2:]

        s_out[:,5*n2:6*n2,0:-s*2,s*2:] = hw[:,5*n2:6*n2,s*2:,:-s*2]
        s_out[:,6*n2:7*n2,0:-s*2,s:] = hw[:,6*n2:7*n2,s*2:,:-s]
        s_out[:,7*n2:8*n2,0:-s*2,0:] = hw[:,7*n2:8*n2,s*2:,:]
        s_out[:,8*n2:9*n2,0:-s*2,0:-s] = hw[:,8*n2:9*n2,s*2:,s:]
        s_out[:,9*n2:10*n2,0:-s*2,0:-s*2] = hw[:,9*n2:10*n2,s*2:,s*2:]

        s_out[:,10*n2:11*n2,s:,s*2:] = hw[:,10*n2:11*n2,  :-s,:-s*2]
        s_out[:,11*n2:12*n2,s:,0:-s*2] = hw[:,11*n2:12*n2,:-s,s*2:]
        s_out[:,12*n2:13*n2,:,s*2:] = hw[:,12*n2:13*n2,  :,:-s*2]
        s_out[:,13*n2:14*n2,:,0:-s*2] = hw[:,13*n2:14*n2,:,s*2:]
        s_out[:,14*n2:15*n2,0:-s,s*2:] = hw[:,14*n2:15*n2,  s:,:-s*2]
        s_out[:,15*n2:16*n2,0:-s,0:-s*2] = hw[:,15*n2:16*n2,s:,s*2:]
        s_out[:,16*n2+0*n1:16*n2+1*n1,s:,s:] = hw[:,16*n2+0*n1:16*n2+1*n1,:-s,:-s]
        s_out[:,16*n2+1*n1:16*n2+2*n1,s:,0:] = hw[:,16*n2+1*n1:16*n2+2*n1,:-s,:]
        s_out[:,16*n2+2*n1:16*n2+3*n1,s:,0:-s] = hw[:,16*n2+2*n1:16*n2+3*n1,:-s,s:]
        s_out[:,16*n2+3*n1:16*n2+4*n1,:,s:] = hw[:,16*n2+3*n1:16*n2+4*n1,:,:-s]
        s_out[:,16*n2+4*n1:16*n2+5*n1,:,0:-s] = hw[:,16*n2+4*n1:16*n2+5*n1,:,s:]
        s_out[:,16*n2+5*n1:16*n2+6*n1,0:-s,s:] = hw[:,16*n2+5*n1:16*n2+6*n1,s:,:-s]
        s_out[:,16*n2+6*n1:16*n2+7*n1,0:-s,0:] = hw[:,16*n2+6*n1:16*n2+7*n1,s:,:]
        s_out[:,16*n2+7*n1:16*n2+8*n1,0:-s,0:-s] = hw[:,16*n2+7*n1:16*n2+8*n1,s:,s:]
        return s_out 
    def channel_shift(self, x, div=2, reverse=False):
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        y1 = x.view(1,B*C,H,W)
        y1 = torch.roll(y1, slice_c,1).view(B,C,H,W)
        kernel_size = 5
        if reverse == False:
            y = torch.cat((x[0:1], y1[1:]), dim=0)
            hw = y[:,0:8*self.number,...]
            # other = y[:,slice_c:,...]
        else:
            y = torch.cat((y1[0:-1],x[-1:]), dim=0)
            hw = y[:,-8*self.number:,...]
            # other = y[:,0:-slice_c,...]
        # hw_shifts = self.shift_conv1(hw)
        hw = self.spatial_shift2(hw)
        # for _ in range(kernel_size):
        #     hw = self.shift_conv(hw)
        # print(torch.sum(torch.abs(hw_shifts-hw)), torch.mean(hw_shifts))
        # exit(0)
        # y_hw_shifts = torch.cat((y, hw_shifts), dim=1)
        # exit(0)
        return torch.cat((y, hw), dim=1)

    def forward(self, x, reverse=0):
        x = self.channel_shift(x)
        x = self.encoder_level1(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_1(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_2(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_3(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_4(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_5(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_6(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_7(x)
        return x
class Encoder2(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder2, self).__init__()
        n_feat = n_features
        scale_unetfeats = 0
        act = nn.PReLU()
        n_feat0 = 24
        # n_feats = 48
        self.act = act
        self.encoder_level1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level1_1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level2_1 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level3 = CAB(n_feat + scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level3_1 = CAB(n_feat + scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act)

        # self.encoder_level1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        # self.encoder_level1_1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        #self.encoder_level1_2 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        #self.encoder_level2 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        #self.encoder_level2_1 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        #self.encoder_level3 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        # self.encoder_level3_1 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        # self.encoder_level3 = Encoder_shift_block(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias)
        # self.encoder_level3_1 = Encoder_shift_block(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias)

        # self.concat = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.concat = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.down01 = nn.Sequential(nn.Conv2d(n_feat0, n_feat, 2, 2, 0, bias=False), nn.PReLU())
        # self.act = act
        # self.concat2= conv(n_feat, n_feat, kernel_size, bias=bias)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.decoder_level1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_2 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level2 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level2_1 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        self.decoder_level3 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        self.decoder_level3_1 = Encoder_shift_block(n_feat+scale_unetfeats, kernel_size, reduction, bias)
        # self.decoder_level3 = Encoder_shift_block(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias)
        # self.decoder_level3_1 = Encoder_shift_block(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.upsample0 = PixelShufflePack(n_feat, n_feat0, 2, upsample_kernel=3)
        self.skip_conv = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) #conv(n_feat, n_feat, kernel_size, bias=bias)
        self.out_conv = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.conv_hr0 = conv(n_feat0*2, n_feat0, kernel_size, bias=True)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        div = 4
        self.slice_c =  n_feat // div
    def channel_shift(self, x, div=2, reverse=False):
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        # for kk in range(1, B):
        #     x[kk,-slice_c:] = x[kk-1,-slice_c:]
        y = x.view(1,B*C,H,W)
        return torch.roll(y, slice_c,1).view(B,C,H,W)
    def forward(self, x, reverse=False):
        # shortcut = x 
        # x = self.concat(x)
        x = self.concat(x)
        shortcut = x 
        x = self.down01(x)
        
        enc1 = self.encoder_level1(x)
        enc11 = self.encoder_level1_1(enc1)
        # enc11 = self.encoder_level1_2(enc11)
        enc1_down = self.down12(enc11)
        enc2 = self.encoder_level2(enc1_down)
        enc22 = self.encoder_level2_1(enc2)
        enc2_down = self.down23(enc22)
        enc3 = self.encoder_level3(enc2_down)
        enc33 = self.encoder_level3_1(enc3)
        dec3 = self.decoder_level3(enc33)
        dec33 = self.decoder_level3_1(dec3, reverse=1)
        # x = dec33 + self.skip_attn2(enc22)
        x = self.up32(dec33, self.skip_attn2(enc22))
        dec2 = self.decoder_level2(x)
        dec22 = self.decoder_level2_1(dec2, reverse=1)
        x = self.up21(dec22, self.skip_attn1(enc11))
        # x = dec22 + self.skip_attn1(enc11)
        dec1 = self.decoder_level1(x)
        dec11 = self.decoder_level1_1(dec1, reverse=1)
        dec11 = self.decoder_level1_2(dec11)
        dec11_out = self.conv_hr0(torch.cat((self.upsample0(dec11), self.skip_conv(shortcut)), dim=1))
        dec11_out = self.out_conv(dec11_out)
        return dec11_out # [enc11, enc22, enc33], [dec11, dec22, dec33]


class Decoder(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Decoder, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        # self.concat = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # self.decoder_level1 += [conv(n_feat, 80, kernel_size, bias=bias)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        

        return [dec1, dec2, dec3]
class TFR_UNet(nn.Module):
    def __init__(self, n_feat0, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(TFR_UNet, self).__init__()
        scale_unetfeats = 12
        self.encoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level3 = [CAB(n_feat0 + 2*scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat0, scale_unetfeats)
        self.down23 = DownSample(n_feat0+scale_unetfeats, scale_unetfeats)

        self.decoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level3 = [CAB(n_feat0+scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat0, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat0 + scale_unetfeats, scale_unetfeats)
    def forward(self, x):
        shortcut = x 
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1


@BACKBONES.register_module()
class GShiftNet(nn.Module):

    def __init__(self, n_features=48, future_frames=1, past_frames=1):
        super(GShiftNet, self).__init__()
        # self.para = para
        self.n_feats = n_features
        # n_feat = n_features
        self.n_feats2 = 80
        self.num_ff = future_frames
        self.num_fb = past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.n_feats0 = 24

        self.feat_extract = nn.Sequential(nn.Conv2d(4, self.n_feats0, 3, 1, 1),
            CAB(self.n_feats0, 3, 4, bias=False, act=nn.PReLU()))
        self.conv_last = conv(self.n_feats0, 4, 5, bias=False) 
        self.conv_trans = conv(self.n_feats0, self.n_feats0, 3, bias=True)
        self.lrelu = nn.PReLU() 
        self.stage1 = Encoder2(self.n_feats2)
        self.orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.orb2 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.orb3 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        # self.orb4 = 
        self.orb4 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.orb5 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rorb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rorb2 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rorb3 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        # self.orb4 = 
        self.rorb4 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rorb5 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rconcat = nn.Conv2d(self.n_feats0*3, self.n_feats0, 3, 1, 1, bias=True)
        
        div = 4
        self.slice_c =  self.n_feats // div
    def stage0(self, x0):
        shortcut = x0
        x0 = self.orb1(x0)
        x0 = self.orb2(x0)
        x0 = self.orb3(x0)
        x0 = self.orb4(x0)
        res0 = self.orb5(x0)
        res0 = res0 + shortcut
        return res0, self.conv_trans(res0) # , stage0_img
    def stage2(self, x0, sam1_feats, decoder_out0):
        x = self.rconcat(torch.cat((x0, sam1_feats, decoder_out0), dim=1))
        shortcut = x 
        x = self.rorb1(x)
        x = self.rorb2(x)
        x = self.rorb3(x)
        x = self.rorb4(x)
        x = self.rorb5(x)
        x = x + shortcut
        x = self.conv_last(x)
        return x

    def forward(self, x, k1=None, k2=None, k3=None):
        batch_size, frames, channels, height, width = x.shape
        x = x[0]
        shortcut = x
        x0 = self.feat_extract(x)
        sam_features0, sam_features = self.stage0(x0) 
        decoder_outs = self.stage1(sam_features) 
        #output_features = self.stage2(x0[self.num_fb:frames-self.num_ff], sam_features0[self.num_fb:frames-self.num_ff], decoder_outs[self.num_fb:frames-self.num_ff])
        
        output_features = self.stage2(x0, sam_features0, decoder_outs)
        #print('output_fea.shape=',output_features.shape)
        #print('shortcut.shape=',shortcut.shape)
        output = output_features + shortcut
        out = output.view(batch_size, frames, -1, height, width)

        return out

    def init_weights(self, pretrained=None, strict=False):
            """Init weights for models.

            Args:
                pretrained (str, optional): Path for pretrained weights. If given
                    None, pretrained weights will not be loaded. Default: None.
                strict (bool, optional): Whether strictly load the pretrained
                    model. Default: True.
            """

            if isinstance(pretrained, str):
                logger = get_root_logger()
                logger.info(f"Init weights: {pretrained}")
                load_checkpoint(self, pretrained, strict=strict, logger=logger)
            #elif self.feat_extract.init_cfg is not None:
            #    self.feat_extract.init_weights()

            elif pretrained is not None:
                raise TypeError(f'"pretrained" must be a str or None. '
                                f'But received {type(pretrained)}.')