# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
from paddle import nn, fluid

from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible

__all__ = ['MUnet']


class MUnet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 scale=1,
                 **kwargs):
        super(MUnet, self).__init__()
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')

        #inplanes = make_divisible(inplanes * scale)
        i = 0
        self.block1 = ResidualUnit(
                    in_channels=16,
                    mid_channels=make_divisible(16),
                    out_channels=make_divisible(16),
                    kernel_size=3,
                    stride=(2, 1),
                    use_se=True,
                    act='relu',
                    name='conv' + str(i + 2))
        i = i + 1
        self.block2 = ResidualUnit(
            in_channels=16,
            mid_channels=make_divisible(72),
            out_channels=make_divisible(24),
            kernel_size=3,
            stride=(2, 1),
            use_se=False,
            act='relu',
            name='conv' + str(i + 2))
        i = i + 1

        self.block3 = ResidualUnit(
            in_channels=24,
            mid_channels=make_divisible(88),
            out_channels=24,
            kernel_size=3,
            stride=1,
            use_se=False,
            act='relu',
            name='conv' + str(i + 2))
        i = i + 1

        self.block4 = ResidualUnit(
            in_channels=24,
            mid_channels=make_divisible(96),
            out_channels=40,
            kernel_size=5,
            stride=(2, 1),
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block5 = ResidualUnit(
            in_channels=40,
            mid_channels=make_divisible(240),
            out_channels=40,
            kernel_size=5,
            stride=1,
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block6 = ResidualUnit(
            in_channels=40,
            mid_channels=make_divisible(120),
            out_channels=48,
            kernel_size=5,
            stride=1,
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        # up sampling
        self.up1 = fluid.dygraph.nn.Conv2DTranspose(num_channels=48,
                                         num_filters=24,
                                         filter_size=2,
                                         stride=(2, 1),
                                         dilation=(1, 0),
                                         padding=(0, 0))

        self.block7 = ResidualUnit(
            in_channels=48,
            mid_channels=make_divisible(240),
            out_channels=32,
            kernel_size=5,
            stride=1,
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.up2 = fluid.Conv2DTranspose(num_channels=32,
                                         num_filters=16,
                                         filter_size=2,
                                         stride=(2, 1),
                                         dilation=(1, 0),
                                         padding=(0, 0))

        self.block8 = ResidualUnit(
            in_channels=32,
            mid_channels=make_divisible(120),
            out_channels=40,
            kernel_size=5,
            stride=(2, 1),
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block9 = ResidualUnit(
            in_channels=40,
            mid_channels=make_divisible(120),
            out_channels=48,
            kernel_size=5,
            stride=(2, 1),
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block10 = ResidualUnit(
            in_channels=48,
            mid_channels=make_divisible(144),
            out_channels=48,
            kernel_size=5,
            stride=1,
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block11 = ResidualUnit(
            in_channels=48,
            mid_channels=make_divisible(288),
            out_channels=96,
            kernel_size=5,
            stride=(2, 1),
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1

        self.block12 = ResidualUnit(
            in_channels=96,
            mid_channels=make_divisible(576),
            out_channels=96,
            kernel_size=5,
            stride=1,
            use_se=True,
            act='hardswish',
            name='conv' + str(i + 2))
        i = i + 1
        self.conv2 = ConvBNLayer(
            in_channels=96,
            out_channels=make_divisible(576),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(576)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        y1 = self.block1(x)
        # print(y1.shape)
        x = self.block2(y1)
        #print(x.shape)
        y2 = self.block3(x)
        # print(y2.shape)
        x = self.block4(y2)
        #print(x.shape)
        x = self.block5(x)
        #print(x.shape)
        x = self.block6(x)
        #print(x.shape)

        # upsampling
        x = self.up1(x)
        #print(x.shape)
        x = paddle.concat([x, y2], axis=1)
        #print(x.shape)
        x = self.block7(x)
        #print(x.shape)

        # upsampling
        x = self.up2(x)
        #print(x.shape)
        x = paddle.concat([x, y1], axis=1)
        #print(x.shape)
        x = self.block8(x)
        #print(x.shape)

        x = self.block9(x)
        #print(x.shape)
        x = self.block10(x)
        #print(x.shape)
        x = self.block11(x)
        #print(x.shape)
        x = self.block12(x)
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        return x