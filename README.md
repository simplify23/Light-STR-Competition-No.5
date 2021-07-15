

## 简介
这里是[Paddle轻量级文字识别技术创新大赛](https://aistudio.baidu.com/aistudio/competition/detail/75)终审第5名（5/1364）的代码链接。
- 我们的模型**总大小9.7M A榜精度80.78% B榜精度79%**
- 模型整体pipline：U-mobilenet + downsample + transformer*2
- 最终的提交代码训练细节为： 400轮训练lr0.001 + 150轮训练lr0.00001（去掉mix_up和cutout）
- 模型:[训练模型](https://pan.baidu.com/s/1PgqwvWKTwrKF2icZZGtsSQ)提取码：rezn
- 我们算法的最大特点：**仅仅从模型设计的角度思考了这一问题**，
暂时没有使用模型压缩的策略（如剪枝，蒸馏，量化等，未来会考虑）

## 目录
- [算法流程](#算法介绍)
- [数据增强](#数据增强)
- [环境部署](#环境部署)
- [如何运行](#如何运行)
- [训练策略](#训练策略)
> [模型链接](https://pan.baidu.com/s/1PgqwvWKTwrKF2icZZGtsSQ) 提取码：rezn
> + model_400为第一步训练的模型（提供动态图）
> + model_400_150 为第二步训练的模型 （提供静态图+动态图）
> + 最终提交为model_400_150的静态图模型

## 算法介绍
所有的参数设计的实验均记录在[paddle文字识别参数对比实验](https://mbqx5nqmwj.feishu.cn/docs/doccnYUPssndhRB48xR1657ZEMe)中，时间关系并没有及时整理。
### 1. U-mobilenet：
基于mobilenet-large进行改进，参考U-net的结构，在mobilenet中增加了U-net的设计，并将最低层的特征图（j=4）与mobilenet的输出进行concat，实验证明这是一种有效策略(concant部分为j=0 -> j=5; j=2 -> j=7 ; j=4 -> mobilenet的最后输出。)
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
### 2. downsample： 
**这是一种简洁但非常高效的策略**，不同于CRNN本身通过直接压缩到（1,80,640）的形式进行2D向1D的转化，downsample使用高维的卷积信息（高维卷积代码写在U-mobilenet上），在高维卷积的基础上使用池化，再通过降维卷积下降至序列模型需要的通道维度，实验证明，
  - 高维卷积再降维的方式：80通道的序列模型可以达到512的通道的序列模型的精度
  - 池化的使用:让降维卷积大大缩小了参数量（降维卷积只需1 * 1即可）。实验证明，池化比起直接使用3 * 3卷积的方式，并不会造成模型的精度明显下降
> **down sample** 整体流程如下
  > 1.  模型在u-mobilenet阶段，last_conv输出为4 * 320 * 640（W,H,C）
  > 2. 通过最大池化（2,2）-> (2 * 160 * 640)
  > 3. 通过最大池化（2,2）->(1 * 80 * 640)
  > 4. 降维卷积  640-> 80
  > 5. 将模型调整成(B * 1 * 80 * 80)->(B,80,80) (batch, width, channels)

  
### 3. transformer序列模块： 
这里使用了transformer的encoder结构，参考[SRN](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html)我们增加了字符阅读顺序。并使用dim = 80。序列模型部分，我们只使用了2层transformer的encoder结构（事实上一层仅需要0.5M模型大小），实验证明，更多层的transformer可以取得更好的性能。
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
### 1. cutout：由于训练集中含有遮挡样本，所以我们cutout来增强我们的数据，对这部分样本可以起到较好的学习效果。我们采用了几种不同的策略进行cutout增强
        - Random Erasing：传统的cutout策略，随机在图片中选个区域进行cut
        - Single Char Erasing：先根据字符数量对image进行划分，然后在单个子区域进行cut
        - Random：单个子区域随机
        - Crop：单个子区域左右部分cut
        - Line Erasing：模拟线形状遮挡
- 在配置文件Train/dataset/transforms/RecAug下，添加use_cutout作为打开开关，后接bool  
![image](https://user-images.githubusercontent.com/42465965/124563454-a1f8b080-de72-11eb-9768-e5b168af0320.png)
- 效果如下图        
![image](https://user-images.githubusercontent.com/42465965/124558928-b1c1c600-de6d-11eb-8ef8-bb3ab48d15e1.png)
- 代码链接：https://github.com/simplify23/Ultra_light_OCR_No.11/blob/5c0e5259dee28484f68122b9955f2469300b89eb/ppocr/data/imaug/cut_aug.py
### 2. mixup：将随机的两张样本按比例混合，计算loss按比例分配
- 配置文件Global下，添加mix_up，后接概率(对每个batch以一定概率是否使用mix up)  
- 代码链接：https://github.com/simplify23/Ultra_light_OCR_No.11/blob/5c0e5259dee28484f68122b9955f2469300b89eb/tools/program.py#L245
- 参考论文：https://arxiv.org/abs/1903.04246

### 3. srn resize：将输入图像按照长宽比例进行不同的resize，在统一padding到统一尺寸
- 在配置文件Train/dataset/transforms下，添加SRNRecResizeImgEx字段，后接reszie尺寸，如[ 3, 64, 640 ]、[ 3, 32, 320]
 ![image](https://user-images.githubusercontent.com/42465965/124562638-d7e96500-de71-11eb-8a6c-8d5172e64007.png)

- 参考论文：https://arxiv.org/abs/2009.09941
## 参数调优与设置
- max_text_length：从原先的25调整到35
- 输入图像的尺寸从原先3 x 32 x 320 调整到 3 x 64 x 640

## 代码整体目录结构
由于篇幅原因，我们将其他数据增强部分，参数优化设置，以及整体代码目录均放在下面文档中：
- [模型的其他策略介绍](./doc/doc_ch/tree.md)

## 环境部署
- **我们的环境部署和PPOCR完全一致**，环境尽可能按官方环境来（值得注意的是python3.8环境似乎不支持U-mobilenet的代码内容）
- python =3.7
- PaddlePaddle-gpu = 2.0.2
#### docker环境
docker运行细节请见文档[Docker化部署](./deploy/docker/hubserving/README_cn.md)。
```
#切换至Dockerfile目录
cd deploy/docker/hubserving/gpu

#生成镜像
docker build -t paddleocr:gpu .

#运行镜像
sudo nvidia-docker run -dp 8868:8868 --name paddle_ocr paddleocr:gpu
```
## 如何运行
### step1 数据准备：请自行下载比赛数据集或参考[训练文件](https://github.com/simplify23/PaddleOCR/blob/release/2.1/doc/doc_ch/recognition.md )
如果需要自定义，请一并修改配置文件
- 训练集路径：
```
|-dataset
  |-train
    |- labeltrain.txt    #标签
    |- Train_000000.jpg
    |- Train_000001.jpg
    |- ...
```
- 验证集路径：
- **我们使用训练集的图片制作了验证集**，制作代码见```utils/process_label.py```
- 验证集的标签我们放到```dataset/val/labelval.txt```
```
|-dataset
  |-val
    |- labelval.txt
    |- ...
```
- 测试集路径：
```
|-dataset
  |-test
    |- Test_000000.jpg
    |- ...
```

### step2: 启动训练
```
cd Ultra_light_OCR_No.11
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c configs/rec/ztl_config_exp/ztl_400_umv_trans_2_9.8M.yml
```
### step3: 启动续训
```
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c configs/rec/ztl_config_exp/ztl_400_umv_trans_2_9.8M_150.yml
```
### step4: 导出模型
```
python3 tools/export_model.py -c output/rec/final_400_umv_trans_2_9.8M_150/config.yml -o Global.pretrained_model=output/rec/final_400_umv_trans_2_9.8M_150/best_accuracy Global.save_inference_dir=./inference/rec/final_400_umv_trans_2_9.8M_150/best_accuracy
```
### step5：预测
```
python3 tools/infer/predict_rec.py --rec_algorithm=CRNN --image_dir="dataset/test" --max_text_length=35 --rec_model_dir=inference/rec/final_400_umv_trans_2_9.8M_150/best_accuracy --rec_char_dict_path=ppocr/utils/ppocr_keys_v2.txt --rec_save_path="inference/rec/final_400_umv_trans_2_9.8M_150/best_accuracy/predict_rec.txt" --use_srn_resize=True --rec_image_shape="3, 64, 640"
```

## 训练策略
### step1  
> + lr 0.01
> + 数据增强概率0.4，打开cutout
> + 打开mixup
> + 训练400 epoch,取best acc的模型
### step2
> + 取上一步的best acc模型，进行续训，配置如下
> + lr 0.00001
> + 数据增强概率0.4
> + 关闭mixup和cutout
> + 续训150epoch，取best acc模型
## License
This project is released under <a href="https://github.com/simplify23/Ultra_light_OCR_No.11/blob/master/LICENSE">Apache 2.0 license</a>


