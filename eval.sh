# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints=pretrain_models/rec_mv3_none_bilstm_ctc_v2.0_train/best_accuracy
