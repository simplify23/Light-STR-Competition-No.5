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
from paddle import nn,fluid

from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible

__all__ = ['UMobileNetV3']


class UMobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=1,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(UMobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16,16, False, 'relu', large_stride[0]],   #8*80 //16*160
                [3, 64, 16,24, False, 'relu', (large_stride[1], 1)], #8*80 //16*160 i=1
                [3, 72, 24,24, False, 'relu', 1],
                [5, 72, 24,40, True, 'relu', (large_stride[2], 1)], #
                [5, 120, 40,40, True, 'relu', 1],
                [5, 120, 40,40, True, 'relu', 1],
                [3, 240, 40,80, False, 'hardswish', 1],
                [3, 200, 80,80, False, 'hardswish', 1],
                [3, 184, 80,80, True, 'hardswish', 1],
                [3, 184, 80,80, True, 'hardswish', 1],
                [3, 480, 80,112, True, 'hardswish', 1],
                [3, 672, 112,112, True, 'hardswish', 1],
                [5, 672, 112,160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160,160, True, 'hardswish', 1],
                [5, 960, 160,160, True, 'hardswish', 1],
            ]
            cfg_u_net = [
                # k,exp,i_c,o_c,se,  act,   s
                [3, 120, 24, 40, True, 'relu', 1],                   #i = 0
                [3, 200, 40, 80, True, 'relu', (large_stride[1], 1)],
                # [3, 240, 80, 80, True, 'hardswish', 1],
                [3, 480, 80, 112, True, 'hardswish', 1],         #i=2
                # [3, 672, 112, 112, True, 'hardswish', 1],
                [3, 672, 112, 160, True, 'hardswish', (large_stride[3], 1)], # i = 3
                # [3, 960, 160,160, True, 'hardswish', 1],
                # up block
                # [3, 672, 240*2,160, True, 'hardswish', 1], # i = 7 concat=240+240
                [3, 480, 160, 112, True, 'hardswish', 1],
                # [3, 480, 112, 112, True, 'hardswish', 1],  #i = 8
                [3, 200, 80*2,80, True, 'hardswish', 1],         #i=5 up1
                [3, 120, 80,  40, True, 'relu', 1],
                [3, 120, 56,24, True, 'relu', 1],        #i=7 up2
            ]
            cls_ch_squeeze = 960
        elif model_name == "large-v2":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16,16, False, 'relu', large_stride[0]],   #8*80 //16*160
                [3, 64, 16,24, False, 'relu', (large_stride[1], 1)], #8*80 //16*160 i=1
                [3, 72, 24,24, False, 'relu', 1],
                [5, 72, 24,40, True, 'relu', (large_stride[2], 1)], #
                [5, 120, 96,40, True, 'relu', 1],
                [5, 120, 40,40, True, 'relu', 1],
                [3, 240, 40,80, False, 'hardswish', 1],
                [3, 200, 80*2,80, True, 'hardswish', 1],
                [3, 184, 80,80, True, 'hardswish', 1],
                [3, 184, 80,80, True, 'hardswish', 1],
                [3, 480, 80,112, True, 'hardswish', 1],
                [3, 672, 112,112, True, 'hardswish', 1],
                [5, 672, 112,160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160,160, True, 'hardswish', 1],
                [5, 960, 160,160, True, 'hardswish', 1],
            ]
            cfg_u_net = [
                # k,exp,i_c,o_c,se,  act,   s
                [3, 120, 24, 40, False, 'relu', 1],                   #i = 0
                [3, 200, 40, 80, True, 'relu', (large_stride[1], 1)],
                # [3, 240, 80, 80, True, 'hardswish', 1],
                [3, 480, 80, 112, True, 'hardswish', 1],         #i=2
                # [3, 672, 112, 112, True, 'hardswish', 1],
                [3, 672, 112, 160, True, 'hardswish', (large_stride[3], 1)], # i = 3
                # [3, 960, 160,160, True, 'hardswish', 1],
                # up block
                # [3, 672, 240*2,160, True, 'hardswish', 1], # i = 7 concat=240+240
                [3, 480, 160, 112, True, 'hardswish', 1],
                # [3, 480, 112, 112, True, 'hardswish', 1],  #i = 8
                [3, 200, 80*2,80, True, 'hardswish', 1],         #i=5 up1
                [3, 120, 80,  40, True, 'relu', 1],
                [3, 120, 56,24, True, 'relu', 1],        #i=7 up2
            ]
            cls_ch_squeeze = 960
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.5]
        assert scale in supported_scale, \
            "nows just supported scales  == 0.5 "

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')
        i = 0
        block_list = []
        ublock_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, in_c,out_c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=make_divisible(scale * in_c),
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * out_c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name='conv' + str(i + 2)))
            inplanes = make_divisible(scale * out_c)
            i += 1
        i = 0
        for (k, exp, i_c,o_c, se, nl, s) in cfg_u_net:
            ublock_list.append(
                ResidualUnit(
                    in_channels= make_divisible(scale * i_c),
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * o_c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name='u-conv' + str(i + 2)))
            i += 1
        self.blocks = nn.Sequential(*block_list)
        self.ublocks = nn.Sequential(*ublock_list)
        self.up1 = nn.Conv2DTranspose(in_channels=make_divisible(scale * 112),
                                        out_channels=make_divisible(scale * 80),
                                        kernel_size=3,
                                        stride=(2, 1),
                                        padding='SAME')
        self.up2 = nn.Conv2DTranspose(in_channels=make_divisible(scale * 40),
                                         out_channels=make_divisible(scale * 24),
                                         kernel_size=3,
                                         stride=(2, 1),
                                         padding='SAME')

        # self.out_channels = inplanes*8 # origin umobilenet
        self.out_channels = inplanes*6
        self.conv2 = ConvBNLayer(
            in_channels=inplanes*2,
            out_channels= self.out_channels, # make_divisible(scale * cls_ch_squeeze), #160 # inplanes*8,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')
        # self.conv_smooth = ConvBNLayer(
        #     in_channels=inplanes*8,
        #     out_channels= self.out_channels, #make_divisible(scale * cls_ch_squeeze), #160
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     groups=1,
        #     if_act=True,
        #     act='hardswish',
        #     name='conv_custom')
        # self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        use_m_concat = False
        u1_concat,u2_concat,u3_concat = None,None,None
        m1_concat,m2_concat,m3_concat = None,None,None
        for i,n in enumerate(self.blocks):
            # mobilenet pipline
            x = n(x)
            # print(i)
            if i==1:
                # u-net pipline
                for j,u in enumerate(self.ublocks):
                    if j == 0:
                        u1_concat = x
                    elif j == 2:
                        u2_concat = x
                    elif j == 4:
                        u3_concat = x
                    elif j == 5:
                        m1_concat = x
                        x = self.up1(x)
                        x = paddle.concat([x, u2_concat], axis=1)
                    elif j == 6:
                        m2_concat = x
                    elif j == 7:
                        m3_concat = x
                        x = self.up2(x)
                        x = paddle.concat([x, u1_concat], axis=1)
                    x = u(x)
            if use_m_concat == True:
                if i == 3:
                    x = paddle.concat([x,m3_concat],axis=1)
                elif i == 6:
                    x = paddle.concat([x,m2_concat],axis=1)
        x = paddle.concat([x,u3_concat],axis=1)
        x = self.conv2(x)
        # x = self.pool(x)
        # x = self.conv_smooth(x)
        # print(x.shape)
        return x
