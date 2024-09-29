import os.path as osp
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class MM_dataset(Dataset):
    def __init__(self, data_dir, transform=None, split='training'):
        super(MM_dataset, self).__init__()

        assert data_dir != None, "training data is None"

        split_num = 500

        if split == 'training':
            self.frameA_list = sorted(glob(osp.join(data_dir, "train", "2", "frameA", "*.png")))[:-split_num]
            self.frameB_list = sorted(glob(osp.join(data_dir, "train", "3", "frameB", "*.png")))[:-split_num]
            self.frameC_list = sorted(glob(osp.join(data_dir, "train", "4", "frameC", "*.png")))[:-split_num]
            self.amp_list = sorted(glob(osp.join(data_dir, "train", "1", "amplified", "*.png")))[:-split_num]

            mag_factor_file = osp.join(data_dir, "train", "train_mf.txt")
            self.mag_factor_list = []
            f = open(mag_factor_file, "r")
            while True: 
                line = f.readline()
                if not line: break
                self.mag_factor_list.append(line)
            f.close()

            self.mag_factor_list = self.mag_factor_list[:-split_num]
            self.transform = transform

        else:

            self.frameA_list = sorted(glob(osp.join(data_dir, "train", "2", "frameA", "*.png")))[-split_num:]
            self.frameB_list = sorted(glob(osp.join(data_dir, "train", "3", "frameB", "*.png")))[-split_num:]
            self.frameC_list = sorted(glob(osp.join(data_dir, "train", "4", "frameC", "*.png")))[-split_num:]
            self.amp_list = sorted(glob(osp.join(data_dir, "train", "1", "amplified", "*.png")))[-split_num:]

            mag_factor_file = osp.join(data_dir, "train", "train_mf.txt")
            self.mag_factor_list = []
            f = open(mag_factor_file, "r")
            while True: 
                line = f.readline()
                if not line: break
                self.mag_factor_list.append(line)
            f.close()

            self.mag_factor_list = self.mag_factor_list[-split_num:]
            self.transform = transform

        assert len(self.frameA_list) == len(self.frameB_list), "The number of data pair is wrong"
        assert len(self.frameA_list) == len(self.frameC_list), "The number of data pair is wrong"
        assert len(self.frameA_list) == len(self.amp_list), "The number of data pair is wrong"
        assert len(self.frameB_list) == len(self.frameC_list), "The number of data pair is wrong"
        assert len(self.frameB_list) == len(self.amp_list), "The number of data pair is wrong"
        assert len(self.frameC_list) == len(self.amp_list), "The number of data pair is wrong"
        assert len(self.frameA_list) == len(self.mag_factor_list), "The number of data pair is wrong"

        print("number of data is %d" % len(self.frameA_list))

    def __getitem__(self, index):

        amp = np.array(Image.open(self.amp_list[index]), dtype=np.float32) / 127.5 - 1.0
        A = np.array(Image.open(self.frameA_list[index]), dtype=np.float32) / 127.5 - 1.0
        B = np.array(Image.open(self.frameB_list[index]), dtype=np.float32) / 127.5 - 1.0
        C = np.array(Image.open(self.frameC_list[index]), dtype=np.float32) / 127.5 - 1.0
        mag_factor = np.array(self.mag_factor_list[index], dtype = 'float32')

        sample = {'amplified': amp, 'frameA': A, 'frameB': B, 'frameC': C, 'mag_factor': mag_factor}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frameA_list)
    