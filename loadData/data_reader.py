import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


HSI_FILES = {
    "Houston_2013": ("Houston_HSI.mat", "Houston"),
    "Augsburg": ("data_HS_LR.mat", "data_HS_LR"),
    "Berlin": ("data_HS_LR.mat", "data_HS_LR"),
}

GT_FILES = {
    "Houston_2013": ("Houston_GT.mat", "Houston_GT"),
    "Augsburg": ("gt.mat", "gt"),
    "Berlin": ("gt.mat", "gt"),
    "TRLabel": ("TrainImage.mat", "TrainImage"),
    "TSLabel": ("TestImage.mat", "TestImage"),
}

AUX_FILES = {
    "Houston_2013": ("Houston_LiDAR.mat", "LiDAR"),
    "Augsburg": ("data_SAR_HR.mat", "data_SAR_HR"),
    "Berlin": ("data_SAR_HR.mat", "data_SAR_HR"),
}


def load_data(dataset, path_data, type_data):
    file_name, key = HSI_FILES[dataset]
    data = sio.loadmat(path_data + file_name)[key].astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    gt_name, gt_key = GT_FILES[dataset if type_data == dataset else type_data]
    data_gt = sio.loadmat(path_data + gt_name)[gt_key].astype(np.int64)
    return data, data_gt


def load_data_LiDAR(dataset, path_data_LiDAR):
    file_name, key = AUX_FILES[dataset]
    return sio.loadmat(path_data_LiDAR + file_name)[key]


def apply_PCA(data, num_components=32):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca
