# recommended paddle.__version__ == 2.0.0
#visul tools
visualdl --logdir="./vdl" --host=172.18.30.123 --port=16006
#run train
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '5,6,7'  tools/train.py -c configs/rec/ztl_config_exp/ztl_mv3_tps_bilstm_ctc.yml
#run quant
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '2,3,4'  tools/slim/quant.py -c configs/rec/ztl_config_exp/ztl_quant.yml
#infer images for pdstates
python3 tools/infer_rec.py -c configs/rec/pp_chinese_ztl_baseline.yml -o Global.pretrained_model=inference/rec/ztl_baseline/inference Global.load_static_weights=false Global.infer_img=train_data/ppdataset/test/testimages
#infer for infer model
python3 tools/infer/predict_rec.py --rec_algorithm=CRNN\
                                   --image_dir="train_data/ppdataset/test/testimages" \
                                   --rec_model_dir=./inference/rec/ztl_500_tps_bilstm_ctc_iter_epoch_480 \
                                   --rec_char_dict_path=ppocr/utils/ppocr_keys_v2.txt\
                                   --rec_save_path="inference/rec/ztl_500_tps_bilstm_ctc_iter_epoch_480/predict_rec.txt"
