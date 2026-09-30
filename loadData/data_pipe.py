import random

import numpy as np
import torch
import yaml

from loadData import data_reader
from loadData.split_data import HyperX, sample_gt


def set_deterministic(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_labels(gt, pad_width):
    labels = []
    for i in range(pad_width, gt.shape[0] - pad_width):
        for j in range(pad_width, gt.shape[1] - pad_width):
            if gt[i][j] > 0:
                labels.append(gt[i][j])
    return labels


def get_data(path_config):
    config = yaml.load(open(path_config, "r"), Loader=yaml.FullLoader)
    dataset_name = config["data_input"]["dataset_name"]
    path_data = config["data_input"]["path_data"]
    path_data_LiDAR = config["data_input"]["path_data_LiDAR"]
    patch_size = config["data_input"]["patch_size"]
    split_type = config["data_split"]["split_type"]
    train_num = config["data_split"]["train_num"]
    train_ratio = config["data_split"]["train_ratio"]
    num_components = config["data_transforms"]["num_components"]
    batch_size = config["data_transforms"]["batch_size"]
    remove_zero_labels = config["data_transforms"]["remove_zero_labels"]

    data, data_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data=dataset_name)
    data, pca = data_reader.apply_PCA(data, num_components=num_components)
    data = data.astype(np.float32)
    data = (data - data.mean(axis=(0, 1), keepdims=True)) / (data.std(axis=(0, 1), keepdims=True) + 1e-8)
    pad_width = patch_size // 2
    img = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))
    img = img[:, :, pad_width:img.shape[2] - pad_width]

    data_LiDAR = data_reader.load_data_LiDAR(dataset_name, path_data_LiDAR=path_data_LiDAR).astype(np.float32)
    if data_LiDAR.ndim == 3:
        data_LiDAR = (data_LiDAR - data_LiDAR.mean(axis=(0, 1), keepdims=True)) / (
            data_LiDAR.std(axis=(0, 1), keepdims=True) + 1e-8)
    else:
        data_LiDAR = (data_LiDAR - data_LiDAR.mean()) / (data_LiDAR.std() + 1e-8)
    img_LiDAR = np.pad(data_LiDAR, pad_width=pad_width, mode="constant", constant_values=(0))
    if len(img_LiDAR.shape) == 3:
        img_LiDAR = img_LiDAR[:, :, pad_width:img_LiDAR.shape[2] - pad_width]

    if split_type == 'number':
        gt = np.pad(data_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        train_gt, test_gt = sample_gt(gt, train_num=train_num, train_ratio=train_ratio, mode=split_type)
    elif split_type == 'disjoint':
        _, train_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data="TRLabel")
        _, test_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data="TSLabel")
        train_gt = np.pad(train_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        test_gt = np.pad(test_gt, pad_width=pad_width, mode="constant", constant_values=(0))
    else:
        raise ValueError("split_type must be 'number' or 'disjoint'")

    train_label = get_labels(train_gt, pad_width)
    test_label = get_labels(test_gt, pad_width)

    train_dataset = HyperX(img, img_LiDAR, train_gt, patch_size=patch_size, flip_augmentation=True,
                           radiation_augmentation=False, mixture_augmentation=False,
                           remove_zero_labels=remove_zero_labels)
    test_dataset = HyperX(img, img_LiDAR, test_gt, patch_size=patch_size, flip_augmentation=False,
                          radiation_augmentation=False, mixture_augmentation=False,
                          remove_zero_labels=remove_zero_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_label, test_label
