import math
import torch
import torch.nn as nn

'''
The pytorch version of Mobilenetv2

'''


def conv_bn(in_c, out_c, stride):
    layer = nn.Sequential(nn.Conv1d(in_c, out_c, stride, 1, bias=False),
                          nn.BatchNorm1d(out_c),
                          nn.ReLU(True))
    return layer


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()

        self.blockname = None
        self.stride = stride
        assert stride in [1, 2], "the stride of mobilenet must be 1 or 2"
        self.res_connect = self.stride == 1 and in_c == out_c
        self.base_stage = nn.Sequential(nn.Conv1d(in_c, in_c * expand_ratio,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=False),
                                        nn.BatchNorm1d(in_c * expand_ratio),
                                        nn.ReLU(True),

                                        nn.Conv1d(in_c * expand_ratio,
                                                  in_c * expand_ratio,
                                                  kernel_size=3,
                                                  stride=stride,
                                                  padding=1,
                                                  groups=in_c * expand_ratio,
                                                  bias=False),
                                        nn.BatchNorm1d(in_c * expand_ratio),
                                        nn.ReLU(True),

                                        nn.Conv1d(in_c * expand_ratio, out_c,
                                                  kernel_size=1, stride=1,
                                                  padding=0, bias=False))
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        short_cut = x
        if self.res_connect:
            out = self.base_stage(x) + short_cut
        else:
            out = self.base_stage(x)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, in_c, num_cls, T, width_ratio):
        super().__init__()

        self.interverted_residual_setting = [
                # t, c, n, s
                # t: expand_ratio
                # c: used for computing output channel
                # n: stacked stage
                # s: stride
                [1, 16, 1, 1],
                [T, 24, 2, 1],
                [T, 32, 3, 2],
                [T, 64, 4, 2],
                [T, 96, 3, 1],
                [T, 160, 3, 2],
                [T, 320, 1, 1],
            ]

        input_channel = int(32 * width_ratio)
        self.conv1 = nn.Sequential(nn.Conv1d(in_c, input_channel, kernel_size=25, stride=2, padding=1, bias=False),
                                   nn.BatchNorm1d(input_channel),
                                   nn.ReLU(True))
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_ratio)
            layers = []
            strides = [s] + [1] * (n-1)
            for stride in strides:
                layers += [InvertedResidual(input_channel, output_channel, stride, t)]
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channels = int(1280 * width_ratio) if width_ratio > 1 else 1280
        self.conv2 = nn.Sequential(nn.Conv1d(input_channel, self.last_channels, kernel_size=1,
                                             stride=1, padding=0, bias=False),
                                   nn.BatchNorm1d(self.last_channels),
                                   nn.ReLU(True))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.last_channels, num_cls)


    def forward(self, x):
        b = x.shape[0]
        x = self.conv1(x)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        x = self.blocks[4](x)
        x = self.blocks[5](x)
        x = self.blocks[6](x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(b, -1)
        x = self.fc(x)

        return x

def mobilenet_half(in_c, num_cls):
    model = MobileNetV2(in_c=in_c, num_cls=num_cls, T=6, width_ratio=0.5)
    return model


if __name__ == "__main__":
    model = mobilenet_half(in_c=1, num_cls=8)
    tensor = torch.rand(2, 1, 1024)
    print(model(tensor).shape)