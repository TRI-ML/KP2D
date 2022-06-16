# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob

from PIL import Image
from torch.utils.data import Dataset

class SonarSimLoader(Dataset):
    """
    Sonar Simulator dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, root_dir,noise_util, data_transform=None):

        super().__init__()
        self.root_dir = root_dir

        self.files=[]
        self.noise_util = noise_util
        for filename in glob.glob(root_dir + '/**/*.jpg'):
            self.files.append(filename)
        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _read_gray_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, idx):

        filename = self.files[idx]
        image = self._read_gray_file(filename)

        if image.mode == 'L':
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {'image': image_new, 'idx': idx}
        else:
            sample = {'image': image, 'idx': idx}


        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
