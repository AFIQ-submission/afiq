# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Optional

import torch
import torchvision

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torchprofile import profile_macs

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    if os.path.exists("characterization.csv"):
        os.remove("characterization.csv")
    if os.path.exists("mispredictions.csv"):
        os.remove("mispredictions.csv")

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    with open("acc1_baseline.csv", "r") as f:
        baseline_lines = f.readlines()

    batch_index = 0
    accum_batch_size = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with open ("mlp_macs.txt", "a") as f:
            f.write("new batch" + "\n")
        # macs = profile_macs(model, images)
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # with open("acc1_baseline.csv", 'a') as f:
        #     f.write(f"{acc1.item()}\n")
        #     for i in range(target.shape[0]):
        #         f.write(f"{target[i].item()},{pred[i][0].item()}\n")

        batch_size = images.shape[0]

        if 0:
            for i in range(batch_size):
                baseline_line = baseline_lines[batch_index + accum_batch_size + i + 1]
                target, baseline_pred = baseline_line.split(",")
                if int(target) == int(baseline_pred) and int(baseline_pred) != int(pred[i][0]):
                    torchvision.utils.save_image(images[i], f"mispredicted_images_oracle/image_{accum_batch_size + i}.png")
                    with open("mispredictions_oracle.csv", 'a') as f:
                        f.write(f"{accum_batch_size + i},{int(target)},{int(baseline_pred)},{int(pred[i][0])},\n")
                if int(target) != int(baseline_pred) and int(target) == int(pred[i][0]):
                    torchvision.utils.save_image(images[i], f"improved_predicted_images_oracle/image_{accum_batch_size + i}.png")
                    with open("improved_predictions_oracle.csv", 'a') as f:
                        f.write(f"{accum_batch_size + i},{int(target)},{int(baseline_pred)},{int(pred[i][0])},\n")


        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        batch_index += 1
        accum_batch_size += batch_size
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if os.path.exists("mispredictions.csv"):
        with open("mispredictions.csv", 'r') as neg_values_file:
            lines = neg_values_file.readlines()
            total_values = 0
            precision_sum = 0.0
            recall_sum = 0.0
            negative_predictive_value_sum = 0.0
            true_negative_rate_sum = 0.0
            false_positive_rate_sum = 0.0
            false_negative_rate_sum = 0.0
            proportion_pred_neg_sum = 0.0
            threshold = 0.0
            single_value_prediction = 0.0
            num_bits_to_calculate = 0.0
            neg_values_sum = 0.0

            for line in lines:
                neg_values,precision,recall,negative_predictive_value,true_negative_rate,false_positive_rate,false_negative_rate,proportion_pred_neg,total_value,threshold,single_value_prediction,num_bits_to_calculate = [float(x) for x in line.split(",")]
                neg_values_sum += neg_values
                precision_sum += precision*total_value
                recall_sum += recall*total_value
                negative_predictive_value_sum += negative_predictive_value*total_value
                true_negative_rate_sum += true_negative_rate*total_value
                false_positive_rate_sum += false_positive_rate*total_value
                false_negative_rate_sum += false_negative_rate*total_value
                proportion_pred_neg_sum += proportion_pred_neg*total_value
                total_values += total_value
            with open("results.csv", 'a') as results_file:
                results_file.write(f"{neg_values_sum/total_values},{precision_sum/total_values},{recall_sum/total_values},{negative_predictive_value_sum/total_values},{true_negative_rate_sum/total_values},{false_positive_rate_sum/total_values},{false_negative_rate_sum/total_values},{proportion_pred_neg_sum/total_values},{threshold},{single_value_prediction},{num_bits_to_calculate},")

    with open("results.csv", 'a') as results_file:
        results_file.write(f"{metric_logger.acc1.global_avg:.3f},\n")
    log_dir = str(os.environ['log_dir'])
    with open(f"{log_dir}/results.csv", 'a') as results_file:
        results_file.write(f"{metric_logger.acc1.global_avg:.3f},\n")
    bits = int(os.environ['bits'])
    topk_percent = float(os.environ['topk_percent'])
    threshold = float(os.environ['threshold'])
    kqa_threshold = float(os.environ['kqa_threshold'])
    kqa_topk_prop = float(os.environ['kqa_topk_prop'])
    with open("sum_results.csv", 'a') as results_file:
        results_file.write(f"{bits},{topk_percent},{threshold},{kqa_topk_prop},{kqa_threshold}, {metric_logger.acc1.global_avg:.3f},")
    with open("kqa_results.csv", 'a') as results_file:
        results_file.write(f"{bits},{topk_percent},{threshold},{kqa_topk_prop},{kqa_threshold}, {metric_logger.acc1.global_avg:.3f},")
    with open("kqa_sum_results.csv", 'a') as results_file:
        results_file.write(f"{bits},{topk_percent},{threshold},{kqa_topk_prop},{kqa_threshold}, {metric_logger.acc1.global_avg:.3f},")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
