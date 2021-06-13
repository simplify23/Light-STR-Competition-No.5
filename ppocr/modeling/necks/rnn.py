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
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer
from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from ppocr.modeling.heads.self_attention import WrapEncoderForFeature, MixerPatch


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
        # self.mixer = MixerPatch(
        #     d_hidden=in_channels,
        #     d_hidden_out=self.out_channels,
        #     d_patch=self.patch,
        #     d_patch_scale=2,
        #     d_patch_out= self.patch//4,
        #     prepostprocess_dropout=0.1,
        #     relu_dropout=0.1)
        # self.pool = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        # print(x.shape)
        x = self.pool(x)
        conv_out = self.conv1(x)
        B, C, H, W = conv_out.shape
        # C = 240
        # H = 1
        # W = 80
        conv_out = paddle.reshape(x=conv_out, shape=[-1, self.out_channels, self.patch])
        # conv_out = conv_out.squeeze(axis=2)
        conv_out = conv_out.transpose([0, 2, 1])

        # mixer_out = self.mixer(x)
        # mixer(in-out)  b w*h c
        # out = conv_out+mixer_out # (NTC)(batch, width, channels)
        out = conv_out
        return out,H*W

class TransformerPosEncoder(nn.Layer):
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
            dim_seq = self.width)

    def forward(self, conv_features):
        # feature_dim = conv_features.shape[1]
        feature_dim = self.width
        encoder_word_pos = paddle.to_tensor(np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64'))
        enc_inputs = [conv_features, encoder_word_pos, None]
        out = self.wrap_encoder_for_feature(enc_inputs)
        return out

class EncoderWithTrans(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithTrans, self).__init__()
        self.out_channels = hidden_size *2 # 3 #2
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
        if img2seq == 'cnn+mixer':
            self.encoder_reshape = Im2Seq_downsample(in_channels,hidden_size,self.patch)
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
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size,num_layers,self.patch)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type!= 'cnn':
            x,self.patch = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x