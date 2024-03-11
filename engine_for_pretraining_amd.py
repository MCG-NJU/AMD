# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_one_epoch(args,model_teacher: torch.nn.Module,model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model_teacher.eval()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    loss_func = nn.MSELoss()
    if args.align_loss=='l2':
        loss_func_distil = nn.MSELoss()
    if args.align_loss=='l1':
        loss_func_distil = nn.L1Loss()
    if args.align_loss=='sml1':
        loss_func_distil = nn.SmoothL1Loss()
        
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos_student, bool_masked_pos_teacher, bool_masked_pos_diff = bool_masked_pos
        bool_masked_pos_student = bool_masked_pos_student.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_masked_pos_teacher = bool_masked_pos_teacher.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_masked_pos_diff = bool_masked_pos_diff.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_student].reshape(B, -1, C)
            
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                dir_teacher_feat, gen_teacher_feat = model_teacher(images, (bool_masked_pos_student ,bool_masked_pos_diff))

            outputs, dir_student_feat, gen_student_feat = model(images, (bool_masked_pos_student, bool_masked_pos_diff, bool_masked_pos_teacher))
            loss_mae = loss_func(input=outputs, target=labels)
            loss_distil_sum_dir = 0.0
            loss_distil_sum_gen = 0.0
            for idx in range(len(dir_student_feat)):
                loss_distil_sum_dir += loss_func_distil(dir_student_feat[idx], dir_teacher_feat[idx])
            loss_distil_dir = loss_distil_sum_dir / len(dir_student_feat)
            for idx in range(len(gen_student_feat)):
                loss_distil_sum_gen += loss_func_distil(gen_student_feat[idx], gen_teacher_feat[idx])
            loss_distil_gen = loss_distil_sum_gen / len(gen_student_feat)
            loss = args.recfac * loss_mae + args.dirfac * loss_distil_dir + args.genfac * loss_distil_gen
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(loss_distil_dir=loss_distil_dir.item())
        metric_logger.update(loss_distil_gen=loss_distil_gen.item())
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss_mae=loss_mae.item(), head="loss")
            log_writer.update(loss_distil_dir=loss_distil_dir.item(), head="loss")
            log_writer.update(loss_distil_gen=loss_distil_gen.item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
