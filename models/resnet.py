import torch
from torch import nn
import numpy as np
import math

from models.DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class GtAttn(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.attn_dim = 32
        self.num_class = num_class
        self.proj1 = nn.Linear(4, self.attn_dim)
        self.proj2 = nn.Conv2d(64, self.attn_dim*num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x, a, mask):
        '''
        x (B, num_class, 3, 4, H, W)
        a (B, 64, H, W)
        mask (B, num_class, H, W, 3)
        '''

        B, _, H, W = a.size()
        x = x.permute(0, 1, 4, 5, 2, 3) # (B, N, H, W, 3, 4)
        x = self.proj1(x) # (B, num_class, H, W, 3, attn_dim)

        a = self.proj2(a) # (B, self.attn_dim*num_class, H, W)
        a = a.view(B, self.num_class, self.attn_dim, H, W)  # (B, num_class, attn_dim, H, W)
        a = a.permute(0, 1, 3, 4, 2) # (B, num_class, H, W, attn_dim)
        a = torch.unsqueeze(a, -1) # (B, num_class, H, W, attn_dim, 1)
        
        score = torch.matmul(x, a).squeeze(-1) # (B, num_class, H, W, 3)
        score = score.masked_fill(mask==0, -1e9)
        score = torch.nn.functional.softmax(score.div(self.attn_dim**0.5), dim=-1)
        
        return score

class CenterNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        super(CenterNet, self).__init__()
        
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers1 = self._make_deconv_layer(256, 4)
        self.deconv_layers2 = self._make_deconv_layer(128, 4)
        self.deconv_layers3 = self._make_deconv_layer(64, 4)

        # self._make_head(256, head_conv, 1)
        # self._make_head(128, head_conv, 2)
        self._make_head(64, head_conv, 3)

        self.attn = GtAttn(8)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        padding, output_padding = self._get_deconv_cfg(num_kernels)

        fc = DCN(
            self.inplanes, num_filters, 
            kernel_size=(3,3), stride=1,
            padding=1, dilation=1, deformable_groups=1
        )

        upsample = nn.ConvTranspose2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=num_kernels,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=self.deconv_with_bias
        )
        fill_up_weights(upsample)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(num_filters, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(upsample)
        layers.append(nn.BatchNorm2d(num_filters, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))

        self.inplanes = num_filters

        return nn.Sequential(*layers)
    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return padding, output_padding
     
    def _make_head(self, plane, head_conv, branch_id):
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(plane, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
                if 'seg' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)

            else:
                fc = nn.Conv2d(plane, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'seg' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head+str(branch_id), fc)

    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> Loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)
        
        print('=> Init deconv weights from normal distribution')
        for name, m in self.deconv_layers1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layers2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layers3.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, gt_margin, mask, phase):
        y = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers1(x)
        y.append(x)
        # print(x.size())
        x = self.deconv_layers2(x)
        y.append(x)
        # print(x.size())
        x = self.deconv_layers3(x)
        y.append(x)
        # print(x.size())
        
        ret = {}
        
        for i in range(2, 3):
            for head in self.heads:
                h = head + str(i+1)
                if head == 'seg':
                    ret[h] = _sigmoid(self.__getattr__(h)(y[i]))
                elif head == 'margin':
                    ret[h] = self.__getattr__(h)(y[i]).exp()
                else:
                    ret[h] = self.__getattr__(h)(y[i])

        if phase == 'train' or phase == 'val':
            score = self.attn(gt_margin, y[2], mask) # (B, num_class, H, W, 3)
            ret['score'] = score

        return ret


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_centernet(num_layers, heads, head_conv=64):
    resnet_spec = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }

    block_class, layers = resnet_spec[num_layers]
    model = CenterNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers)
    return model

if __name__ == "__main__":
    heads = {'margin': 32, 'seg': 8}
    model = get_centernet(18, heads)
    
    x = torch.rand(2, 3, 512, 512)
    y = model(x)
    for k, v in y.items():
        print(k, v.size())