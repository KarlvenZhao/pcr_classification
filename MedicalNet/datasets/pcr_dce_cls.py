'''
Dataset for PCR classification using post DCE volumes.
'''

import os
import random

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy import ndimage


class PcrDceClsDataset(Dataset):
    def __init__(self, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f if line.strip()]
        print("Processing {} datas".format(len(self.img_list)))
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        ith_info = self.img_list[idx].split()
        img_name = ith_info[0]
        label = int(ith_info[1])
        if not os.path.isfile(img_name):
            raise FileNotFoundError(img_name)

        img = nib.load(img_name)
        if img is None:
            raise RuntimeError("Failed to load image: {}".format(img_name))

        img_array = self.__data_process__(img)
        img_array = self.__nii2tensorarray__(img_array)

        return img_array, label

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x]).astype("float32")
        return new_data

    def __load_array__(self, img):
        # NIfTI is usually (H, W, D); transpose to (D, H, W)
        arr = np.asanyarray(img.dataobj)
        if arr.ndim != 3:
            raise ValueError("Expected 3D volume, got shape {}".format(arr.shape))
        arr = arr.transpose(2, 0, 1)
        return arr

    def __drop_invalid_range__(self, volume):
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        if non_zeros_idx[0].size == 0:
            return volume

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __random_center_crop__(self, data):
        nonzeros = np.where(data != data.flat[0])
        if nonzeros[0].size == 0:
            return data

        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(nonzeros), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(nonzeros), axis=1)

        Z_min = random.randint(0, max(0, min_D))
        Y_min = random.randint(0, max(0, min_H))
        X_min = random.randint(0, max(0, min_W))

        Z_max = random.randint(min(img_d, max_D + 1), img_d)
        Y_max = random.randint(min(img_h, max_H + 1), img_h)
        X_max = random.randint(min(img_w, max_W + 1), img_w)

        if Z_max <= Z_min or Y_max <= Y_min or X_max <= X_min:
            return data

        return data[Z_min:Z_max, Y_min:Y_max, X_min:X_max]

    def __center_crop__(self, data):
        d, h, w = data.shape
        td, th, tw = self.input_D, self.input_H, self.input_W
        if d <= td and h <= th and w <= tw:
            return data

        z0 = max(0, (d - td) // 2)
        y0 = max(0, (h - th) // 2)
        x0 = max(0, (w - tw) // 2)
        z1 = min(d, z0 + td)
        y1 = min(h, y0 + th)
        x1 = min(w, x0 + tw)

        return data[z0:z1, y0:y1, x0:x1]

    def __itensity_normalize_one_volume__(self, volume):
        pixels = volume[volume > 0]
        if pixels.size == 0:
            return volume.astype("float32")

        mean = pixels.mean()
        std = pixels.std() if pixels.std() > 0 else 1.0
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [
            self.input_D * 1.0 / depth,
            self.input_H * 1.0 / height,
            self.input_W * 1.0 / width,
        ]
        data = ndimage.zoom(data, scale, order=0)
        return data

    def __data_process__(self, img):
        data = self.__load_array__(img)
        data = self.__drop_invalid_range__(data)

        if self.phase == "train":
            data = self.__random_center_crop__(data)
        else:
            data = self.__center_crop__(data)

        data = self.__resize_data__(data)
        data = self.__itensity_normalize_one_volume__(data)
        return data
