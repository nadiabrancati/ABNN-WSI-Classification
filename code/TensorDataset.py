"""
Created on July 2021

@author: Nadia Brancati

"""
import torch.utils.data as data
import torch
import os
import glob
from torchvision import transforms
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator


class TensorDataset(data.Dataset):

    def __init__(self, root_dir, extension):
        self.root_dir=root_dir
        self.ext=extension
        self.classes, self.class_to_idx = self.find_classes()
        self.file_list = glob.glob(self.root_dir + "**/*."+self.ext)

    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        tensor_U = torch.load(file_name,map_location=torch.device("cpu"))
        tensor_U = tensor_U.squeeze()
        tensor_U = tensor_U.unsqueeze(0)
        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.Tensor([self.classes.index(running_label)],device=torch.device("cpu")).to(torch.int64)
        return (tensor_U, label, file_name)

    def __len__(self):
        return len(self.file_list)

class ImageDataset(data.Dataset):

    def __init__(self, root_dir, patch, scale, overlap, device,extension,workers=10,level=16):
        self.root_dir=root_dir
        self.patch=patch
        self.scale=scale
        self.overlap=overlap
        self.workers=workers
        self.ext=extension
        self.data_transform = transforms.Compose([
            transforms.Resize(self.scale),
            transforms.CenterCrop(self.scale),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # standard normalization
        ])
        self.device=device
        self.classes, self.class_to_idx = self.find_classes()
        self.file_list = glob.glob(self.root_dir+"**/*."+self.ext)
        self.level=level

    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        slide = open_slide(os.path.join(self.root_dir,self.file_list[index]))
        tiles = DeepZoomGenerator(slide,tile_size=self.patch,overlap=self.overlap,limit_bounds=False)
        self.level = tiles.level_count-1
        W,H=tiles.level_tiles[self.level]
        #creation of a tensor with size [in_H/patch_size, inW/patch_size] where in_H and in_W are height and width of the original image
        ris = torch.zeros([H, W, 3, 200, 200], device=self.device)

        #arrangement the original images in a set of patch of size
        for w in range(W):
            for h in range(H):
                tile = tiles.get_tile(tiles.level_count-1,(w,h))
                tile_mod = self.data_transform(tile)
                ris[h,w]=tile_mod

        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.tensor([self.classes.index(running_label)], device=self.device)

        return (ris,label, file_name)

    def __len__(self):
        return len(self.file_list)


    def tensor_and_info(self,index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        tensor=torch.load(self.file_list[index])
        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.tensor([self.classes.index(running_label)], device=self.device)
        return (tensor,label,file_name)
