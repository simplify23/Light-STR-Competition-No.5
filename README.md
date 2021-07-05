## 简介
这里是[Paddle轻量级文字识别技术创新大赛](https://aistudio.baidu.com/aistudio/competition/detail/75)第11名的代码链接。
- 我们的模型**总大小9.8M A榜精度80.78% B榜精度79%**
- 模型整体pipline：U-mobilenet + downsample + transformer*2
- 最终的提交代码训练细节为： 400轮训练lr0.001 + 150轮训练lr0.00001（去掉mix_up和cutout）
- 我们算法的最大特点：**仅仅从模型设计的角度思考了这一问题**，
暂时没有使用模型压缩的策略（如剪枝，蒸馏，量化等，未来会考虑）

## 算法介绍
所有的参数设计的实验均记录在[paddle文字识别参数对比实验](https://mbqx5nqmwj.feishu.cn/docs/doccnYUPssndhRB48xR1657ZEMe)中，时间关系并没有及时整理。
- **U-mobilenet**：基于mobilenet-large进行改进，参考U-net的结构，在mobilenet中增加了U-net的设计，并将最低层的特征图（j=4）与mobilenet的输出进行concat，实验证明这是一种有效策略(concant部分为j=0 -> j=5; j=2 -> j=7 ; j=4 -> mobilenet的最后输出。)
```
#mobilenet上的改进
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
                [3, 184, 80,80, True, 'hardswish', 1],     #增加了这一层的SEnet
                [3, 184, 80,80, True, 'hardswish', 1],     #增加了这一层的SEnet
                [3, 480, 80,112, True, 'hardswish', 1],
                [3, 672, 112,112, True, 'hardswish', 1],
                [5, 672, 112,160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160,160, True, 'hardswish', 1],
                [5, 960, 160,160, False, 'hardswish', 1],   #通过敏感度分析，发现这一SEnet并没有用
            ]
```
```
增加了U-net1结构放入mobilenet中,结构模仿mobilenet进行设计，配置文件如下
cfg_u_net = [
                # k,exp,i_c,o_c,se,  act,   s
                [3, 120, 24, 40, True, 'relu', 1],                           #i = 0
                [3, 200, 40, 80, True, 'relu', (large_stride[1], 1)],
                [3, 480, 80, 112, True, 'hardswish', 1],                     #i = 2
                [3, 672, 112, 160, True, 'hardswish', (large_stride[3], 1)], # i = 3
                # up block
                [3, 480, 160, 112, True, 'hardswish', 1],
                [3, 200, 80*2,80, True, 'hardswish', 1],                      #i = 5 up1
                [3, 120, 80,  40, True, 'relu', 1],
                [3, 120, 56,24, True, 'relu', 1],                             #i = 7 up2(400,600)
            ]
```
- **downsample**： **这是一种简洁但非常高效的策略**，不同于CRNN本身通过直接压缩到（1,80,640）的形式进行2D向1D的转化，downsample使用高维的卷积信息（高维卷积代码写在U-mobilenet上），在高维卷积的基础上使用池化，再通过降维卷积下降至序列模型需要的通道维度，实验证明，
  - 高维卷积再降维的方式：80通道的序列模型可以达到512的通道的序列模型的精度
  - 池化的使用:让降维卷积大大缩小了参数量（降维卷积只需1 * 1即可）。实验证明，池化比起直接使用3 * 3卷积的方式，并不会造成模型的精度明显下降
> **down sample** 整体流程如下
  > 1.  模型在u-mobilenet阶段，last_conv输出为4 * 320 * 640（W,H,C）
  > 2. 通过最大池化（2,2）-> (2 * 160 * 640)
  > 3. 通过最大池化（2,2）->(1 * 80 * 640)
  > 4. 降维卷积  640-> 80
  > 5. 将模型调整成(B * 1 * 80 * 80)->(B,80,80) (batch, width, channels)

  
- **transformer序列模块**： 这里使用了transformer的encoder结构，参考[SRN](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html)我们增加了字符阅读顺序。并使用dim = 80。序列模型部分，我们只使用了2层transformer的encoder结构（事实上一层仅需要0.5M模型大小），实验证明，更多层的transformer可以取得更好的性能。
> 这里的整体流程为： 
> + transformer 2层: d_inner_hid=self.hidden_dims * 2 这里从4改成了2，降低参数量但并没有牺牲精度
> + 升维linear   80->160
```
class EncoderWithTrans(nn.Layer):
    def __init__(self, in_channels, hidden_size,num_layers = 2,patch = 80):
        super(EncoderWithTrans, self).__init__()
        self.out_channels = hidden_size *2 #2
        self.custom_channel = hidden_size
        self.transformer=TransformerPosEncoder(hidden_dims=self.custom_channel, num_encoder_tus=num_layers,width=patch)
        self.up_linear = EncoderWithFC(self.custom_channel,self.out_channels,'up_encoder')

    def forward(self, x):
        x = self.transformer(x)
        x = self.up_linear(x)
        return x
```
## 数据增强
> - 使用了mix up 数据增强策略
> - 使用了Cut out 数据增强策略

由于篇幅原因，我们将数据增强部分，参数优化设置，以及整体代码目录均放在下面文档中：
- [模型的其他策略介绍](https://github.com/simplify23/Ultra_light_OCR_No.11/blob/master/doc/doc_ch/tree.md)

## 环境部署
## 如何运行



