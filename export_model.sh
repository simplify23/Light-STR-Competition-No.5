# recommended paddle.__version__ == 2.0.0
#python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '4,5,6,7'  tools/train.py -c configs/rec/ztl_config_exp/ztl_mv3_none_bilstm4_att.yml
python3 tools/export_model.py -c output/rec/ztl_500_tps_bilstm_ctc_48/config.yml \
                              -o Global.pretrained_model=output/rec/ztl_500_tps_bilstm_ctc_48/iter_epoch_480 \
                               Global.save_inference_dir=./inference/rec/ztl_500_tps_bilstm_ctc_iter_epoch_480
python deploy/slim/quantization/quant.py -c configs/rec/ztl_config_exp/ztl_mv3_tps_bilstm_att.yml \
                                        -o Global.pretrain_weights=output/rec/ztl_500_tps_bilstm4_att_96/best_accuracy \
                                          Global.save_model_dir=./output/quant_model/rare_test
#lite for quant
paddle_lite_opt --model_file=./inference.pdmodel \
      --param_file=./inference.pdiparams\
      --valid_targets=npu \
      --optimize_out_type=protobuf \
      --optimize_out=test_opt