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

## 参数调优与设置

