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

import math

import paddle
from paddle import ParamAttr, nn
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle.fluid as fluid
import numpy as np
gradient_clip = 10


class WrapEncoderForFeature(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 use_pos_module=False,
                 dim_seq = 80,
                 bos_idx=0):
        super(WrapEncoderForFeature, self).__init__()
        t_shape = dim_seq
        self.use_pos_module =use_pos_module
        if use_pos_module == True:
            self.prepare_encoder = PrepareEncoder_ztl(
                src_vocab_size,
                d_model,
                max_length,
                prepostprocess_dropout,
                bos_idx=bos_idx,
                word_emb_param_name="src_word_emb_table")
        else:
            self.prepare_encoder = PrepareEncoder(
                src_vocab_size,
                d_model,
                max_length,
                prepostprocess_dropout,
                bos_idx=bos_idx,
                word_emb_param_name="src_word_emb_table")
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, t_shape, preprocess_cmd,
                               postprocess_cmd,use_pos_module)

    def forward(self, enc_inputs):
        conv_features, src_pos, src_slf_attn_bias = enc_inputs
        if self.use_pos_module == True:
            enc_input,pos_enc = self.prepare_encoder(conv_features, src_pos,src_slf_attn_bias)
        else:
            enc_input = self.prepare_encoder(conv_features, src_pos)
            pos_enc = None
        enc_output = self.encoder(enc_input, src_slf_attn_bias,pos_enc)
        return enc_output


class WrapEncoder(nn.Layer):
    """
    embedder + encoder
    """

    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
        super(WrapEncoder, self).__init__()
        t_shape = 25
        self.prepare_decoder = PrepareDecoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx)
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, t_shape, preprocess_cmd,
                               postprocess_cmd)

    def forward(self, enc_inputs):
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_decoder(src_word, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class Encoder(nn.Layer):
    """
    encoder
    """

    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 t_shape,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 use_pos_module=False):

        super(Encoder, self).__init__()
        atten_method ='MHA'
        self.use_pos_module = use_pos_module
        self.encoder_layers = list()
        test_flag = False
        if test_flag == True:
            for i in range(n_layer): #layer-2
                self.encoder_layers.append(
                    self.add_sublayer(
                        "layer_mlp%d" % i,
                        gmlp_block(n_head, d_key, d_value, d_model, d_inner_hid,
                                     prepostprocess_dropout, attention_dropout,
                                     relu_dropout,  t_shape,atten_method, preprocess_cmd,
                                     postprocess_cmd,i)))
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout,  t_shape,atten_method, preprocess_cmd,
                                 postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self, enc_input, attn_bias,pos_enc=None):
        for encoder_layer in self.encoder_layers:
            # enc_output = encoder_layer(enc_input+pos_enc, attn_bias)
            if self.use_pos_module==True:
                enc_input+=pos_enc
                enc_output = encoder_layer(enc_input, attn_bias)
            else:
                enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output
        enc_output = self.processer(enc_output)
        return enc_output

class gmlp_block(nn.Layer):
    """
    gmlp_block
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 dim_seq=80,
                 atten_method = 'MHA',
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 i = 0):

        super(gmlp_block, self).__init__()
        self.atten_method = atten_method
        self.ff_dim = d_model *2
        self.dim_seq = dim_seq
        self.i = i
        self.act = paddle.nn.GELU()
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.proj_in = paddle.nn.Linear(in_features=d_model, out_features=self.ff_dim)
        self.sgu = SpatialGatingUnit(self.ff_dim,dim_seq=self.dim_seq)
        # if self.i==0:
        #     self.proj_out = paddle.nn.Linear(in_features=self.ff_dim//2, out_features=d_model)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        input = self.preprocesser1(enc_input)
        input = self.act(self.proj_in(input))
        input = self.sgu(input)
        # if self.i ==0:
        #     input = self.proj_out(input)
        ffn_output = self.postprocesser1(enc_input, input)
        return ffn_output

class EncoderLayer(nn.Layer):
    """
    EncoderLayer
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 t_shape,
                 atten_method = 'MHA',
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderLayer, self).__init__()
        self.atten_method = atten_method
        self.test = False
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        if atten_method == 'MHA'and self.test!=True:
            self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        elif self.test == True:
            self.self_attn = MultiHeadAttention_ztl(d_key, d_value, d_model, n_head,
                                                attention_dropout)
        elif atten_method == 'Mixer':
            # 32 is w*h
            self.self_attn = MixerBlock(t_shape*4, t_shape, t_shape, relu_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        input = self.preprocesser1(enc_input)
        if self.atten_method=='MHA':
            attn_output = self.self_attn(input, None, None, attn_bias)
        elif self.atten_method=='Mixer':
            attn_output = self.self_attn(input)
        attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output

class MixerPatch(nn.Layer):
    """
    for img2seq mixerpatch
    """

    def __init__(self,
                 d_hidden,
                 d_hidden_out,
                 d_patch,
                 d_patch_scale,
                 d_patch_out,
                 prepostprocess_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(MixerPatch, self).__init__()
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_hidden,
                                                 prepostprocess_dropout)
        self.ffn = paddle.nn.Linear(
            in_features=d_hidden, out_features=d_hidden_out)
        self.act = paddle.nn.GELU()
        # self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_hidden_out,
        #                                           prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_hidden_out,
                                                 prepostprocess_dropout)
        self.mixer_patch = MixerBlock(d_patch_scale*d_patch, d_patch, d_patch_out, relu_dropout)
        # self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_hidden_out,
        #                                           prepostprocess_dropout)

    def forward(self, enc_input):
        enc_input = self.preprocesser1(enc_input)
        enc_input = self.act(self.ffn(enc_input))
        # attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.mixer_patch(self.preprocesser2(enc_input))
        # ffn_output = self.postprocesser2(ffn_output, enc_input)
        return ffn_output

class MultiHeadAttention(nn.Layer):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.,test=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        if test == True:
            self.d_key = 1
        else:
            self.d_key = d_key
        self.d_query = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = paddle.nn.Linear(
            in_features=d_model, out_features=self.d_query * n_head, bias_attr=False)
        self.k_fc = paddle.nn.Linear(
            in_features=d_model, out_features=self.d_key * n_head, bias_attr=False)
        self.v_fc = paddle.nn.Linear(
            in_features=d_model, out_features=d_value * n_head, bias_attr=False)
        self.proj_fc = paddle.nn.Linear(
            in_features=d_value * n_head, out_features=d_model, bias_attr=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = paddle.reshape(x=q, shape=[0, 0, self.n_head, self.d_query])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = paddle.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            if self.d_key == 1:
                k = paddle.tile(k,[1,1,1,self.d_query])
            k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
            v = paddle.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            v = paddle.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            if static_kv and not "static_k" in cache:
                # for encoder-decoder attention in inference and has not cached
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # for decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
                k = paddle.concat([cache_k, k], axis=2)
                v = paddle.concat([cache_v, v], axis=2)
                cache["k"], cache["v"] = k, v

        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self._prepare_qkv(queries, keys, values, cache)
        # print(q.shape)
        # print(k.shape)
        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = product * self.d_model**-0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product)
        if self.dropout_rate:
            weights = F.dropout(
                weights, p=self.dropout_rate, mode="downscale_in_infer")
        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)

        return out

class MultiHeadAttention_ztl(nn.Layer):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.,test=True):
        super(MultiHeadAttention_ztl, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_query = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = paddle.nn.Linear(
            in_features=d_model, out_features=self.d_query * n_head, bias_attr=False)
        self.s_fc = paddle.nn.Linear(
            in_features=d_model, out_features= n_head, bias_attr=False)
        self.k_fc = paddle.nn.Linear(
            in_features=d_model, out_features=self.d_key * n_head, bias_attr=False)
        self.v_fc = paddle.nn.Linear(
            in_features=d_model, out_features=d_value * n_head, bias_attr=False)
        self.proj_fc = paddle.nn.Linear(
            in_features=d_value * n_head, out_features=d_model, bias_attr=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = paddle.reshape(x=q, shape=[0, 0, self.n_head, self.d_query])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            s = self.s_fc(keys)
            k = paddle.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
            v = paddle.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            v = paddle.transpose(x=v, perm=[0, 2, 1, 3])

            s = paddle.reshape(x=s, shape=[0, 0, self.n_head, 1])
            s = paddle.tile(s,[1,1,1,self.d_query])
            s = paddle.transpose(x=s, perm=[0, 2, 1, 3])

        return q, k, v, s

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v, s = self._prepare_qkv(queries, keys, values, cache)

        s_product = paddle.matmul(x=q, y=s, transpose_y=True)
        s_product = s_product * self.d_model**-0.5
        if attn_bias is not None:
            s_product += attn_bias
        s_weights = F.softmax(s_product)
        if self.dropout_rate:
            s_weights = F.dropout(
                s_weights, p=self.dropout_rate, mode="downscale_in_infer")

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = product * self.d_model**-0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product)

        if self.dropout_rate:
            weights = F.dropout(
                weights, p=self.dropout_rate, mode="downscale_in_infer")
        out = paddle.matmul(weights+s_weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)

        return out

class PrePostProcessLayer(nn.Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y is not None else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(self.sublayers()),
                        paddle.nn.LayerNorm(
                            normalized_shape=d_model,
                            weight_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(1.)),
                            bias_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(0.)))))
            elif cmd == "d":  # add dropout
                self.functors.append(lambda x: F.dropout(
                    x, p=dropout_rate, mode="downscale_in_infer")
                                     if dropout_rate else x)

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class PrepareEncoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareEncoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.src_max_len, embedding_dim=self.src_emb_dim)
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word_emb = src_word
        src_word_emb = fluid.layers.cast(src_word_emb, 'float32')
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        src_pos = paddle.squeeze(src_pos, axis=-1)
        src_pos_enc = self.emb(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        else:
            out = enc_input
        return out

class PrepareEncoder_ztl(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareEncoder_ztl, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.dim = src_emb_dim//8
        self.src_max_len = src_max_len
        self.postion_aware_module = Postion_Aware_Module(n_head=8,d_key=self.dim, d_value=self.dim, d_model=src_emb_dim, d_inner_hid=src_emb_dim*2)
        self.attention = MultiHeadAttention(n_head=8,d_key=self.dim, d_value=self.dim, d_model=src_emb_dim, dropout_rate=0.,test=False)
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.src_max_len, embedding_dim=self.src_emb_dim)
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos,attn_bias):
        src_word_emb = src_word
        Batch_size,P,H = src_word.shape
        src_word_emb = fluid.layers.cast(src_word_emb, 'float32')
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        src_pos = paddle.squeeze(src_pos, axis=-1)
        src_pos_enc = self.emb(src_pos)

        key = self.postion_aware_module(src_word_emb,attn_bias)
        value = src_word_emb
        src_pos_enc = paddle.unsqueeze(src_pos_enc,axis=0)
        query = paddle.tile(src_pos_enc,[Batch_size,1,1])
        # print(query.shape)
        # print(key.shape)
        pos_enc = self.attention(queries=query,keys = key,values=value,attn_bias = attn_bias)
        # enc_input = src_word_emb + pos_enc
        # if self.dropout_rate:
        #     out = F.dropout(
        #         x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        # else:
        #     out = enc_input
        return src_word_emb,pos_enc

class Postion_Aware_Module(nn.Layer):
    def __init__(self,
                 n_head=8,
                 d_key=80,
                 d_value=80,
                 d_model=80,
                 d_inner_hid=80*2,
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(Postion_Aware_Module, self).__init__()
        #block(attention ,CNN)
        self.test = False
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                                attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.cnn = nn.Sequential(
            nn.Conv1D(d_model,d_inner_hid,3,padding=1),
            nn.GELU(),
            nn.Conv1D(d_inner_hid,d_model,3,padding=1)
        )
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        input = self.preprocesser1(enc_input)
        attn_output = self.self_attn(input, None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)
        attn_output = self.preprocesser2(attn_output)
        # B P H
        ffn_output = paddle.transpose(attn_output, perm=[0, 2, 1])
        ffn_output = self.cnn(ffn_output)
        ffn_output = paddle.transpose(ffn_output, perm=[0, 2, 1])
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output

class PrepareDecoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareDecoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        """
        self.emb0 = Embedding(num_embeddings=src_vocab_size,
                              embedding_dim=src_emb_dim)
        """
        self.emb0 = paddle.nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=self.src_emb_dim,
            padding_idx=bos_idx,
            weight_attr=paddle.ParamAttr(
                name=word_emb_param_name,
                initializer=nn.initializer.Normal(0., src_emb_dim**-0.5)))
        self.emb1 = paddle.nn.Embedding(
            num_embeddings=src_max_len,
            embedding_dim=self.src_emb_dim,
            weight_attr=paddle.ParamAttr(name=pos_enc_param_name))
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word = fluid.layers.cast(src_word, 'int64')
        src_word = paddle.squeeze(src_word, axis=-1)
        src_word_emb = self.emb0(src_word)
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        src_pos = paddle.squeeze(src_pos, axis=-1)
        src_pos_enc = self.emb1(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        else:
            out = enc_input
        return out


class FFN(nn.Layer):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = paddle.nn.Linear(
            in_features=d_model, out_features=d_inner_hid)
        self.fc2 = paddle.nn.Linear(
            in_features=d_inner_hid, out_features=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        if self.dropout_rate:
            hidden = F.dropout(
                hidden, p=self.dropout_rate, mode="downscale_in_infer")
        out = self.fc2(hidden)
        return out

class MixerBlock(nn.Layer):
    def __init__(self, d_inner_hid, d_model, d_out, dropout_rate):
        super(MixerBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.act = paddle.nn.GELU()
        self.fc1 = paddle.nn.Linear(
            in_features=d_model, out_features=d_inner_hid)
        self.fc2 = paddle.nn.Linear(
            in_features=d_inner_hid, out_features=d_out)
        self.fctest = paddle.nn.Linear(
            in_features=d_model, out_features=d_out)

    def forward(self, x):
        # x: b w*h c
        x = paddle.transpose(x, perm=[0, 2, 1])
        hidden = self.fctest(x)
        # hidden = self.fc1(x)
        hidden = self.act(hidden)
        if self.dropout_rate:
            hidden = F.dropout(
                hidden, p=self.dropout_rate, mode="downscale_in_infer")
        # hidden = self.fc2(hidden)
        hidden = paddle.transpose(hidden, perm=[0, 2, 1])
        return hidden

class SpatialGatingUnit(nn.Layer):
    def __init__(self, dim, dim_seq):
        super(SpatialGatingUnit, self).__init__()
        dim_out = dim //2
        self.head = 8
        # self.norm = nn.BatchNorm(dim_out)
        self.norm = nn.LayerNorm(
            normalized_shape=dim_out,
            weight_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)))
        self.conv1 = nn.Conv1D(dim_seq,dim_seq,1)
        # self.conv1 = nn.Conv2D(dim_seq,dim_seq,(1,1))
        # self.norm2 = nn.LayerNorm(
        #     normalized_shape=dim_out,
        #     weight_attr=fluid.ParamAttr(
        #         initializer=fluid.initializer.Constant(1.)),
        #     bias_attr=fluid.ParamAttr(
        #         initializer=fluid.initializer.Constant(0.)))
        # self.norm3 = nn.LayerNorm(
        #     normalized_shape=dim_out,
        #     weight_attr=fluid.ParamAttr(
        #         initializer=fluid.initializer.Constant(1.)),
        #     bias_attr=fluid.ParamAttr(
        #         initializer=fluid.initializer.Constant(0.)))

    def forward(self, x):
        B, P, C = x.shape
        # un-exp idea multi-head
        # x = paddle.reshape(x=x, shape=[B, P, self.head, C//self.head])
        # x = paddle.reshape(x=x, shape=[B//self.head, P, self.head,  C])
        res, gate = paddle.chunk(x, chunks=2, axis=-1)
        # res = self.norm2(res)
        gate = self.norm(gate)
        gate = self.conv1(gate)
        out = gate * res
        # out = F.dropout(
        #     out, p=0.1, mode="downscale_in_infer")
        # out = paddle.reshape(x=out, shape=[B, P, C//2])
        # out = self.norm3(out)
        return out
