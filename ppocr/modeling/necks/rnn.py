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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
import paddle
import numpy as np
import paddle.nn.functional as F
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer
from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from ppocr.modeling.heads.self_attention import WrapEncoderForFeature, MixerPatch

class FusionAttention(nn.Layer):
    def __init__(self,in_channel,):
        super(FusionAttention, self).__init__()
        self.fc_down = EncoderWithFC(in_channel,in_channel//4,"fusion_down")
        self.fc_up = EncoderWithFC(in_channel//4,in_channel,"fusion_up")
        self.smooth = EncoderWithFC(in_channel,in_channel,"fusion_smooth")

    def forward(self, x, **kwargs):
        unit = F.relu(self.fc_down(x))
        x = self.smooth(x)
        unit = F.hardsigmoid(self.fc_up(unit))
        return x*unit

class AddAttention(nn.Layer):
    def __init__(self,in_channel,):
        super(AddAttention, self).__init__()
        self.fc_x = EncoderWithFC(in_channel*2,in_channel,"fusion_x1")
        # self.fc_x2 = EncoderWithFC(in_channel,in_channel,"fusion_x2")
        self.smooth = EncoderWithFC(in_channel,in_channel,"fusion_smooth")

    def forward(self, x1,x2, **kwargs):
        f = paddle.concat([x1, x2], axis=2)
        f_att = F.sigmoid(self.fc_x(f))
        output = f_att * x1 + (1-f_att) * x2
        output = self.smooth(output)
        return output

class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x,H*W

class Im2Seq_downsample(nn.Layer):
    def __init__(self, in_channels, out_channels=80, patch=2*160, conv_name='conv_patch',**kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.patch = patch
        if out_channels == 512:
            self.use_resnet = True
        else:
            self.use_resnet = False
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels= self.out_channels, #make_divisible(scale * cls_ch_squeeze), #160
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name=conv_name)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        '''
        模型在u-mobilenet阶段，last_conv输出为4*320*640（W,H,C）
        downsample阶段流程
            1.通过最大池化（2,2）-> (2*160*640)
            2.通过最大池化（2,2）->(1*80*640)
            3.降维卷积  640-> 80
            4.将模型调整成(B*1*80*80)->(B,80,80) (batch, width, channels)
        '''
        x = self.pool(x)
        B, C, H, W = x.shape
        if H != 1 :
            pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            x = pool2(x)
        x = self.conv1(x)

        # C = 240
        # H = 1
        # W = 80
        # print(x.shape)
        assert H == 1
        conv_out = paddle.reshape(x=x, shape=[-1, self.out_channels, self.patch])
        # conv_out = conv_out.squeeze(axis=2)
        conv_out = conv_out.transpose([0, 2, 1])

        # mixer_out = self.mixer(x)
        # mixer(in-out)  b w*h c
        # out = conv_out+mixer_out # (NTC)(batch, width, channels)
        # print(conv_out.shape)
        out = conv_out
        return out,H*W

class Im2Seq_Double(nn.Layer):
    def __init__(self, in_channels, out_channels=80, patch=2*160, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.patch = patch
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels= self.out_channels, #make_divisible(scale * cls_ch_squeeze), #160
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_patch')
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
    def forward(self, enc_out):
        # print(x.shape)
        out = []
        for x in enc_out:
            x = self.pool(x)
            B, C, H, W = x.shape
            if H != 1 :
                pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
                x = pool2(x)
            x = self.conv1(x)
            # print(x.shape)
            assert H == 1
            conv_out = paddle.reshape(x=x, shape=[-1, self.out_channels, self.patch])
            # conv_out = conv_out.squeeze(axis=2)
            conv_out = conv_out.transpose([0, 2, 1])
            out.append(conv_out)

        # (NTC)(batch, width, channels)
        # print(conv_out.shape)
        return out,H*W

class Im2Seq_SqueezePatch(nn.Layer):
    def __init__(self, in_channels, out_channels=80, patch=2*160, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.patch = patch
        x,y = 2, 160
        # x,y = 4, 320
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels,self.out_channels,1,1),
            nn.LayerNorm([self.out_channels,x,y]),
            nn.GELU(),
        )
        self.seq_up = nn.Sequential(
            nn.Conv2D(2,self.out_channels,1,1),
            nn.LayerNorm([self.out_channels,x,y]),
            nn.GELU(),
        )
        self.seq_down = nn.Sequential(
            nn.Conv2D(self.out_channels,1,1,1),
            nn.LayerNorm([1,x,y]),
            # nn.GELU(),
        )
        # self.conv_smooth = ConvBNLayer(
        #     in_channels=self.patch,  # 80
        #     out_channels= self.patch, # 80
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     if_act=True,
        #     act='hardswish',
        #     name='conv_smooth')
        self.avgpool = nn.AdaptiveAvgPool2D((1,self.patch))  #patch = 80

    def forward(self, x):
        # expect c = 80
        # print(p_out.shape)
        p_out = self.conv1(x)

        # attention width door unit
        # B C H W
        avgout = paddle.mean(x, axis=1, keepdim=True)
        maxout= paddle.max(x, axis=1, keepdim=True)
        a_out = paddle.concat([avgout, maxout], axis=1)
        # B 1 H W
        a_out = self.seq_up(a_out)
        # a_out = F.relu(a_out)
        a_out = self.seq_down(a_out)
        a_out = F.softmax(a_out)
        # print(a_out.shape)

        conv_out = a_out*p_out
        conv_out = self.avgpool(conv_out)
        # conv_out = self.conv_smooth(conv_out)
        # conv_out (B,C,H,W)
        B, C, H, W = conv_out.shape
        assert H == 1

        conv_out = conv_out.squeeze(axis=2)
        conv_out = conv_out.transpose([0, 2, 1])
        # print(out.shape)
        # (NTC)(batch, width, channels)
        return conv_out, 80

class TransformerPosEncoder(nn.Layer):
    '''
        序列阶段，这里使用了transformer的encoder结构，代码在transformer encoder的基础上加入了字符阅读顺序（与SRN论文一致）
        由于downsample步骤的缘故，transformer encoder只需要80的通道大小即可获得512通道同样的性能
        1、transformer encoder 2层（经过测试一层参数量0.1M ,模型保存大小约0.5M）
        2、其余配置与SRN论文保持一致
    '''
    def __init__(self,  max_text_length=35, num_heads=8,
                 num_encoder_tus=2, hidden_dims=96,width=80):
        super(TransformerPosEncoder, self).__init__()
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.hidden_dims = hidden_dims
        self.width =width
        # Transformer encoder
        t = 35 #256
        c = 96
        self.wrap_encoder_for_feature = WrapEncoderForFeature(
            src_vocab_size=1,
            max_length=t,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            # channel mix for double 256
            d_inner_hid=self.hidden_dims*2, #2,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True,
            use_pos_module=False,
            dim_seq = self.width)

    def forward(self, conv_features):
        # feature_dim = conv_features.shape[1]
        feature_dim = self.width
        encoder_word_pos = paddle.to_tensor(np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64'))
        enc_inputs = [conv_features, encoder_word_pos, None]
        out = self.wrap_encoder_for_feature(enc_inputs)
        return out

class EncoderWithTrans_Double(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithTrans_Double, self).__init__()
        self.out_channels = hidden_size *2# 3 #2
        self.custom_channel = hidden_size
        self.transformer=TransformerPosEncoder(hidden_dims=self.custom_channel, num_encoder_tus=num_layers,width=patch)
        self.transformer2=TransformerPosEncoder(hidden_dims=self.custom_channel, num_encoder_tus=num_layers,width=patch)
        self.up_linear = EncoderWithFC(self.custom_channel,self.out_channels,'up_encoder')
        self.up_linear2 = EncoderWithFC(self.custom_channel,self.out_channels,'up_encoder2')
        self.attention = AddAttention(self.out_channels)

    def forward(self, x1, x2):
        # x = self.down_linear(x)
        x1 = self.transformer(x1)
        x1 = self.up_linear(x1)

        x2 = self.transformer(x2)
        x2 = self.up_linear(x2)
        # x, _ = self.lstm(x)
        x = self.attention(x1,x2)
        # print(x.shape)
        return x

class EncoderWithTrans(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithTrans, self).__init__()
        self.out_channels = hidden_size *2 #2
        self.custom_channel = hidden_size
        self.transformer=TransformerPosEncoder(hidden_dims=self.custom_channel, num_encoder_tus=num_layers,width=patch)
        # self.down_linear = EncoderWithFC(in_channels,self.custom_channel,'down_encoder')
        self.up_linear = EncoderWithFC(self.custom_channel,self.out_channels,'up_encoder')

    def forward(self, x):
        # x = self.down_linear(x)
        x = self.transformer(x)
        x = self.up_linear(x)
        # x, _ = self.lstm(x)
        # print(x.shape)
        return x

class EncoderWithTransRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithTransRNN, self).__init__()
        self.out_channels = hidden_size *2 # 3 #2
        self.custom_channel = hidden_size *2
        self.use_lstm = False
        self.transformer=TransformerPosEncoder(hidden_dims=hidden_size, num_encoder_tus=num_layers,width=patch)
        # self.down_linear = EncoderWithFC(in_channels,self.custom_channel,'down_encoder')
        self.down_linear = EncoderWithFC(self.custom_channel,hidden_size,'down_encoder')
        self.atten = FusionAttention(self.out_channels)
        if self.use_lstm == True:
            self.lstm = nn.LSTM(in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        # x = self.down_linear(x)
        trans_out = self.transformer(x)
        if self.use_lstm == True:
            rnn_out, _ = self.lstm(x)
            rnn_out = self.down_linear(rnn_out)
        else:
            rnn_out = x

        out = paddle.concat([trans_out,rnn_out],axis=2)
        out = self.atten(out)

        return out

class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=num_layers)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size,name='reduce_encoder_fea',num_layers=1,patch = 80):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels, name=name)#'reduce_encoder_fea')
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name = name)
            # name='reduce_encoder_fea')

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, num_layers=2,patch = 120, img2seq='origin',**kwargs):
        super(SequenceEncoder, self).__init__()
        self.patch = patch
        self.img2seq =img2seq
        if img2seq == 'cnn+mixer':
            self.encoder_reshape = Im2Seq_downsample(in_channels,hidden_size,self.patch)
        elif img2seq == 're-patch':
            self.encoder_reshape = Im2Seq_SqueezePatch(in_channels,hidden_size,self.patch)
        elif img2seq == 'double':
            self.encoder_reshape = Im2Seq_downsample(in_channels,hidden_size,self.patch,'x1_conv_name')
            self.encoder_reshape2 = Im2Seq_downsample(in_channels,hidden_size,self.patch,'x2_conv_name')
        else:
            self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        elif encoder_type == 'cnn':
            self.encoder = ConvBNLayer(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                act='relu',
                name= "downsample")
            self.out_channels = hidden_size
            self.only_reshape = False
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'transformer': EncoderWithTrans,
                'transformer_double': EncoderWithTrans_Double,
                'transformer+rnn': EncoderWithTransRNN,
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size,num_layers,self.patch)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type!= 'cnn' and self.img2seq!='double':
            x, _ = self.encoder_reshape(x)
        elif self.img2seq =='double':
            x[0], _ = self.encoder_reshape(x[0])
            x[1], _ = self.encoder_reshape2(x[1])
        if not self.only_reshape and self.img2seq!='double':
            x = self.encoder(x)
        elif self.img2seq =='double':
            x = self.encoder(x[0],x[1])
        return x