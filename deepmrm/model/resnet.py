from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, conv1x1

def conv1x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 3),
        stride=(1, stride),
        padding=(0, 1),
        groups=groups,
        dilation=dilation,
        bias=False,
    )

class BasicBlock1x3(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


class Bottleneck1x3(Bottleneck):
    expansion: int = 4
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


class ResNet1x3(ResNet):
    def __init__(self, layers=[2, 2, 2, 2], block=BasicBlock1x3, inplanes=64, change_conv1=True):
        super(ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d

        self.inplanes = inplanes
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 64
        planes = [self.inplanes*(2**i) for i in range(len(layers))]

        if change_conv1:
            pre_inplanes = self.inplanes // 2
            self.conv1 = nn.Sequential(
                            nn.Conv2d(2, pre_inplanes, kernel_size=(1, 7), 
                                    stride=(1, 1), padding=(0, 3), 
                                    groups=2, bias=False),
                            self._norm_layer(pre_inplanes),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(pre_inplanes, self.inplanes, kernel_size=(3, 7), 
                                    stride=(1, 2), padding=(0, 3), 
                                    groups=2, bias=False))
        else:
            self.conv1 = nn.Conv2d(2, 
                                self.inplanes, 
                                kernel_size=(3, 7), 
                                stride=(1, 2), 
                                padding=(0, 3), 
                                bias=False)

        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(
                        nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                        nn.AdaptiveAvgPool2d((1, None)))

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[-1] * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

