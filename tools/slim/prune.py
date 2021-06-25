from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import time
__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from paddleslim.dygraph import L1NormFilterPruner
from paddle.static import InputSpec as Input
from paddle.jit import to_static
import paddle
import paddle.distributed as dist

paddle.seed(2)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program
from ppocr.utils.stats import TrainingStats

dist.get_world_size()

def prune(net,img_size=[1,3,32,320]):
    pruner = L1NormFilterPruner(net, img_size)
    pruner.prune_vars({'conv2d_22.w_0':0.5, 'conv2d_20.w_0':0.6}, axis=0)

def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])

    # quant(model)
    quant_config = {
        # weight preprocess type, default is None and no preprocessing is performed.
        'weight_preprocess_type': None,
        # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }
    quanter = QAT(config=quant_config)
    quanter.quantize(model)

    if config['Global']['distributed']:
        model = paddle.DataParallel(model)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train
    quant_train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer,quanter)

def save_model(path,quanter,net):
    net = to_static(
        net,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None,3,32,320], dtype='int8')
        ])
    paddle.jit.save(net, path)
    inputs=[paddle.static.InputSpec(shape=[None, 3, 32,320], dtype='float32')]
    # inputs = [Input([None, 3, 32, 320], 'float32', name='image')]
    # path="./quant_inference_model"
    quanter.save_quantized_model(
        net,
        path,
        input_spec=inputs)

def quant_train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          vdl_writer=None,
          quanter=None):
    cal_metric_during_train = config['Global'].get('cal_metric_during_train',
                                                   False)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']

    global_step = 0
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
                format(start_eval_step, eval_batch_step))

    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()

    use_srn = config['Architecture']['algorithm'] == "SRN"

    if 'start_epoch' in best_model_dict:
        start_epoch = best_model_dict['start_epoch']
    else:
        start_epoch = 1

    for epoch in range(start_epoch, epoch_num + 1):
        train_dataloader = build_dataloader(
            config, 'Train', device, logger, seed=epoch)
        train_batch_cost = 0.0
        train_reader_cost = 0.0
        batch_sum = 0
        total_time = 0.
        count = 0
        max_iter = len(train_dataloader)
        batch_start = time.time()
        for idx, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - batch_start
            if idx >= len(train_dataloader):
                break
            lr = optimizer.get_lr()
            images = batch[0]
            if use_srn:
                others = batch[-4:]
                preds = model(images, others)
                model_average = True
            else:
                preds = model(images)
            loss = loss_class(preds, batch)
            avg_loss = loss['loss']
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            train_batch_cost += time.time() - batch_start
            total_time += train_batch_cost
            count += 1
            avg_time = total_time / count
            eta_seconds = avg_time * (max_iter - idx)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            batch_sum += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {k: v.numpy().mean() for k, v in loss.items()}
            stats['lr'] = lr
            train_stats.update(stats)

            if cal_metric_during_train:  # only rec and cls need
                batch = [item.numpy() for item in batch]
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            if vdl_writer is not None and dist.get_rank() == 0:
                for k, v in train_stats.get().items():
                    vdl_writer.add_scalar('TRAIN/{}'.format(k), v, global_step)
                vdl_writer.add_scalar('TRAIN/lr', lr, global_step)

            if dist.get_rank() == 0 and (
                    (global_step > 0 and global_step % print_batch_step == 0) or
                    (idx >= len(train_dataloader) - 1)):
                logs = train_stats.log()
                strs = 'epoch: [{}/{}], iter: {}, {},eta times: {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ips: {:.5f}'.format(
                    epoch, epoch_num, global_step, logs, eta_string,train_reader_cost /
                                                                    print_batch_step, train_batch_cost / print_batch_step,
                    batch_sum, batch_sum / train_batch_cost)
                logger.info(strs)
                train_batch_cost = 0.0
                train_reader_cost = 0.0
                batch_sum = 0
            # eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0 and dist.get_rank() == 0:
                if model_average:
                    Model_Average = paddle.incubate.optimizer.ModelAverage(
                        0.15,
                        parameters=model.parameters(),
                        min_average_window=10000,
                        max_average_window=15625)
                    Model_Average.apply()
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    use_srn=use_srn)
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                # logger metric
                if vdl_writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            vdl_writer.add_scalar('EVAL/{}'.format(k),
                                                  cur_metric[k], global_step)
                if cur_metric[main_indicator] >= best_model_dict[
                    main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model(
                        path='./best_accuary',
                        quanter=quanter,
                        net = model)
                best_str = 'best metric, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                # logger best metric
                if vdl_writer is not None:
                    vdl_writer.add_scalar('EVAL/best_{}'.format(main_indicator),
                                          best_model_dict[main_indicator],
                                          global_step)
            global_step += 1
            optimizer.clear_grad()
            batch_start = time.time()
        if dist.get_rank() == 0:
            save_model(
                path='./other/last',
                quanter=quanter,
                net = model)
    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if dist.get_rank() == 0 and vdl_writer is not None:
        vdl_writer.close()
    return

if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)