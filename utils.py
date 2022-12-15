import os
from skimage import metrics
from skimage.util import random_noise
import numpy as np


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def tensorboard_scalar(input_value, compute_value, step, bar_name, writer, i, epoch, args, is_ssim=0, is_val=0):
    compute_value += input_value.item()

    if is_val == 0:
        if (i + 1) % 1 == 0:
            if is_ssim == 0:
                writer.add_scalar(bar_name,
                                  compute_value,
                                  epoch * step + i * args.train_batch)
                compute_value = 0.0

            if is_ssim == 1:
                writer.add_scalar(bar_name,
                                  1 - compute_value,
                                  epoch * step + i * args.train_batch)
                compute_value = 0.0

    if is_val == 1:
        if is_ssim == 0:
            writer.add_scalar(bar_name,
                              compute_value,
                              epoch * step + i * args.val_batch)
            compute_value = 0.0

        if is_ssim == 1:
            writer.add_scalar(bar_name,
                              1 - compute_value,
                              epoch * step + i * args.val_batch)
            compute_value = 0.0

    return compute_value


def tensorboard_scalars(input_value_a, compute_value_a,
                        input_value_b, compute_value_b,
                        step,
                        bar_name,
                        data_name_a, data_name_b,
                        writer, i, epoch, args, is_ssim=0, is_val=0):
    compute_value_a += input_value_a.item()
    compute_value_b += input_value_b.item()

    if is_val == 0:
        if (i + 1) % 1 == 0:
            if is_ssim == 0:
                writer.add_scalars(bar_name,
                                   {data_name_a: compute_value_a / args.train_batch,
                                    data_name_b: compute_value_b / args.train_batch},
                                   epoch * step + i * args.train_batch)
                compute_value_a = 0.0
                compute_value_b = 0.0

            if is_ssim == 1:
                writer.add_scalars(bar_name,
                                   {data_name_a: 1 - compute_value_a / args.train_batch,
                                    data_name_b: 1 - compute_value_b / args.train_batch},
                                   epoch * step + i * args.train_batch)
                compute_value_a = 0.0
                compute_value_b = 0.0

    if is_val == 1:
        if (i + 1) % 1 == 0:
            if is_ssim == 0:
                writer.add_scalars(bar_name,
                                   {data_name_a: compute_value_a / args.val_batch,
                                    data_name_b: compute_value_b / args.val_batch},
                                   epoch * step + i * args.val_batch)
                compute_value_a = 0.0
                compute_value_b = 0.0

            if is_ssim == 1:
                writer.add_scalars(bar_name,
                                   {data_name_a: 1 - compute_value_a / args.val_batch,
                                    data_name_b: 1 - compute_value_b / args.val_batch},
                                   epoch * step + i * args.val_batch)
                compute_value_a = 0.0
                compute_value_b = 0.0

    return compute_value_a, compute_value_b


def get_psnr(img_a, img_b):
    img1_np = np.array(img_a)
    img2_np = np.array(img_b)
    psnr = metrics.peak_signal_noise_ratio(img1_np, img2_np)
    return psnr


def get_gaussian_noise(img):
    """
    input: [0, 1] or [-1, 1]
    return: noise pic
    """
    img = random_noise(img, var=0.0005)
    return img
