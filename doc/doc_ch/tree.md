## 整体目录结构

该算法 的整体目录结构介绍如下：

```
├── configs
│   └── rec
│       ├── ch_ppocr_v2.0
│       │   ├── rec_chinese_common_train_v2.0.yml
│       │   └── rec_chinese_lite_train_v2.0.yml
│       ├── ztl_config_exp                                  #配置文件的整体位置
│       │   ├── ztl_mv3_tps_bilstm_ctc_124_final_mv.yml     #最后的配置文件 
│       │   ├── ztl_mv3_tps_bilstm_ctc_124.yml
│       │   ├── ztl_mv3_tps_bilstm_ctc_2.yml
│       │   ├── ztl_mv3_tps_bilstm_ctc_64.yml
│       │   ├── ztl_mv3_tps_bilstm_ctc.yml
│       │   ├── ztl_r50_tps_bilstm_ctc_124.yml
│       │   └── ztl_r50_tps_bilstm_ctc.yml
│       ├── ztl_origin.yml
│       └── ztl_rec_mv3_none_bilstm_ctc.yml
├── ppocr
│   ├── data
│   │   ├── imaug                                   #数据增强部分代码
│   │   │   ├── cut_aug.py
│   │   │   ├── east_process.py
│   │   │   ├── iaa_augment.py
│   │   │   ├── __init__.py
│   │   │   ├── label_ops.py
│   │   │   ├── randaugment.py
│   │   │   ├── random_crop_data.py
│   │   │   ├── rec_img_aug.py
│   │   │   ├── sast_process.py
│   │   │   └── text_image_aug
│   │   │       ├── augment.py
│   │   │       ├── __init__.py
│   ├── modeling
│   │   ├── architectures
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   ├── rec_mobilenet_v3.py
│   │   │   ├── rec_resnet_fpn.py
│   │   │   ├── rec_umobilenet_v3.py            #U-mobilenet的backbone代码位置
│   │   ├── heads
│   │   │   ├── rec_att_head.py
│   │   │   ├── rec_ctc_head.py                 #最后使用的是CTC的结构
│   │   │   └── self_attention.py
│   │   ├── necks
│   │   │   ├── __init__.py
│   │   │   ├── rnn.py                         # downsample+transformer的代码位置
│   └── utils
│       ├── ppocr_keys_v1.txt
│       ├── ppocr_keys_v2.txt                 #新设计了字典，适配训练集
│       └── utility.py
├── tools
│   ├── eval.py
│   ├── export_model.py
│   ├── infer
.  .    .
│   │   ├── predict_rec.py
│   │   └── utility.py
│   ├── infer_rec.py
│   ├── program.py
│   └── train.py
├── train_data
│   └── ppdataset -> ../../dataset/ppdataset
└── train.sh
```
## 数据增强策略
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

### 2. mixup：将随机的两张样本按比例混合，计算loss按比例分配
- 配置文件Global下，添加mix_up，后接概率(对每个batch以一定概率是否使用mix up)  
- 代码链接：https://github.com/simplify23/Ultra_light_OCR_No.11/blob/5c0e5259dee28484f68122b9955f2469300b89eb/tools/program.py#L245
- 参考论文：https://arxiv.org/abs/1903.04246

### 3. srn resize：将输入图像按照长宽比例进行不同的resize，在统一padding到统一尺寸
- 在配置文件Train/dataset/transforms下，添加SRNRecResizeImgEx字段，后接reszie尺寸，如[ 3, 64, 640 ]、[ 3, 32, 320]
 ![image](https://user-images.githubusercontent.com/42465965/124562638-d7e96500-de71-11eb-8a6c-8d5172e64007.png)
- 具体实现：
```
def resize_norm_img_srn(img, image_shape):
    imgC, imgH, imgW = image_shape

    img_black = np.zeros((imgH, imgW))
    im_hei = img.shape[0]
    im_wid = img.shape[1]
    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))
    elif im_wid <= im_hei * 4:
        img_new = cv2.resize(img, (imgH * 4, imgH))
    else:
        img_new = cv2.resize(img, (imgW, imgH))

    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_black[:, 0:img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    row, col, c = img_black.shape
    c = 1

    return np.reshape(img_black, (c, row, col)).astype(np.float32)
```
- 参考论文：https://arxiv.org/abs/2009.09941

