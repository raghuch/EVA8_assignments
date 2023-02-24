import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import albumentations as A
import numpy as np


# Don't use torchvision transforms but use albumentations transforms instead

default_train_transforms = A.Compose([
    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(),
    A.CoarseDropout(1, 16, 16, 1, 16, fill_value=0.473363, mask_fill_value=None),
    A.ToGray()
])

default_test_transforms = A.Compose([
    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
])

assn8_train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40),
    A.RandomCrop(32, 32),
    A.HorizontalFlip(),
    A.CoarseDropout(1, 8,8, 1, 8,fill_value=0.473363, mask_fill_value=None)
])

assn8_test_transforms = A.Compose([
    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
])

class AugmentedCIFAR10(Dataset):
    def __init__(self, img_lst, train=True, train_tfms=default_train_transforms, test_tfms=default_test_transforms):
        super().__init__()
        self.img_lst = img_lst
        self.train = train
        self.transforms = train_tfms

        self.norm = test_tfms

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        img, label = self.img_lst[idx]

        if self.train:
            img = self.transforms(image=np.array(img))["image"]
        else:
            img = self.norm(image=np.array(img))["image"]

        img = np.transpose(img, (2,0,1)).astype(np.float32)

        return torch.tensor(img, dtype=torch.float), label





# SEED = 1
# use_cuda = torch.cuda.is_available()
# torch.manual_seed(SEED)
# if use_cuda:
#     torch.cuda.manual_seed(SEED)


def get_augmented_cifar10_dataset(data_root, train_tfms=default_train_transforms, test_tfms=default_test_transforms, batch_sz=128, shuffle=True):
    trainset = datasets.CIFAR10(data_root, train=True, download=True) #, transform=train_transforms)
    testset = datasets.CIFAR10(data_root, train=False, download=True) #, transform=test_transforms)
    use_cuda = torch.cuda.is_available()
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_sz, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=shuffle, batch_size=64)

    train_loader = torch.utils.data.DataLoader(AugmentedCIFAR10(trainset, train=True, train_tfms=default_train_transforms), **dataloader_args)
    test_loader = torch.utils.data.DataLoader(AugmentedCIFAR10(testset, train=False, test_tfms=default_test_transforms), **dataloader_args)

    return train_loader, test_loader

