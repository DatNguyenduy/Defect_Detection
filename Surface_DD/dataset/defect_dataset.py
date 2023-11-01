import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

# import sys
# sys.path.append("..")
from utils.utils import get_all_item_label_path




class DefectDataset(Dataset):
    def __init__(self,data_dir,img_size=(500,250)) -> None:
        super().__init__()
        self.f_imgs,self.f_labels = get_all_item_label_path(input_dir=data_dir,
                                                            item_suffix='.jpg')
        self.img_width = img_size[1]
        self.img_height = img_size[0]
    def __len__(self):
        return len(self.f_imgs)
    def __getitem__(self, index) -> tuple:
        img_path = self.f_imgs[index]
        label_path = self.f_labels[index]
        img = Image.open(img_path)
        img_array = np.array(img.resize((self.img_width,self.img_height),resample=Image.BICUBIC))/255
        label = Image.open(label_path)
        label_array = np.array(label.resize((self.img_width,self.img_height),resample=Image.BICUBIC))/255

        return (torch.tensor([img_array]),torch.tensor([label_array]))   
