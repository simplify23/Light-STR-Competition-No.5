# recommended paddle.__version__ == 2.0.0
#visul tools
#IoQchBjF
visualdl --logdir="./vdl" --host=172.18.30.123 --port=16006
#share visual
python3 -m paddle.distributed.launch --gpus '3'  tools/train.py -c configs/rec/ztl_config_exp/ztl_mv3_tps_bilstm_ctc.yml
#run train
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '1,2,3,4,5,6,7'  tools/train.py -c configs/rec/ztl_config_exp/ztl_r50_tps_bilstm_ctc.yml

#export model
python3 tools/export_model.py -c output/rec/final_ztl_500_mv_trans_4_960/config.yml \
                              -o Global.pretrained_model=output/rec/final_ztl_500_mv_trans_4_960/best_accuracy \
                               Global.save_inference_dir=./inference/rec/final_ztl_500_mv_trans_4_960/best_accuracy
#infer for infer model
python3 tools/infer/predict_rec.py --rec_algorithm=CRNN\
                                   --image_dir="train_data/ppdataset/test/testimages" \
                                   --rec_image_shape="3, 96, 960"\
                                   --use_srn_resize=True\
                                   --max_text_length=35\
                                   --rec_model_dir=inference/rec/final_ztl_500_mv_trans_4_960/best_accuracy\
                                   --rec_char_dict_path=ppocr/utils/ppocr_keys_v2.txt\
                                   --rec_save_path="inference/rec/final_ztl_500_mv_trans_4_960/best_accuracy/latest_predict_rec.txt"
#count error
python3 tools/infer/predict_rec.py --rec_algorithm=STARNet\
                                   --image_dir="train_data/ppdataset/test/testimages" \
                                   --rec_model_dir=inference/rec/ztl_700_none_bilstm_96_4_ctc/latest \
                                   --rec_char_dict_path=ppocr/utils/ppocr_keys_v2.txt\
                                   --rec_save_path="inference/rec/ztl_700_none_bilstm_96_4_ctc/predict_rec.txt"\
                                   --rec_label_path="train_data/ppdataset/train/labeltrain.txt"

#run quant
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '2,3,4'  tools/slim/quant.py -c configs/rec/ztl_config_exp/ztl_quant.yml
#infer images for pdstates
python3 tools/infer_rec.py -c output/rec/fpn50_srn_2_4_baseline/config.yml -o Global.pretrained_model=output/rec/fpn50_srn_2_4_baseline/best_accuracy Global.load_static_weights=false Global.infer_img=train_data/ppdataset/test/testimages

python3 tools/infer/predict_rec.py --rec_algorithm=CRNN\
                                   --image_dir="dataset/test/testimages" \
                                   --rec_image_shape="3, 64, 640"\
                                   --use_srn_resize=True\
                                   --max_text_length=35\
                                   --rec_model_dir=inference/rec/ztl_500_resnet_fpn_50_trans_4_512_640/best_accuracy\
                                   --rec_char_dict_path=ppocr/utils/ppocr_keys_v2.txt\
                                   --rec_save_path="inference/rec/ztl_500_resnet_fpn_50_trans_4_512_640/best_accuracy/latest_predict_rec.txt"

ssh -o ServerAliveInterval=30 zhengtianlun@172.18.30.124