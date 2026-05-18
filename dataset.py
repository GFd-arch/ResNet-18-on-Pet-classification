# download and initialize dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class OxfordPetsDataset(Dataset):
    def __init__(self, img_size=224, root='./data'):
        self.img_size = img_size
        self.root = root
        self.dataset = OxfordIIITPet(
            root=self.root,
            split='trainval',
            target_types='category',
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

    def get_dataloaders(self, batch_size=32):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), # 改变图像的亮度、对比度、饱和度和色调
            transforms.RandomHorizontalFlip(),          # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25)            # 随机擦除
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        train_dataset = OxfordIIITPet(
            root='./data',
            split='trainval',
            target_types='category',
            download=False,
            transform=train_transform
        )
        test_dataset = OxfordIIITPet(
            root='./data',
            split='test',
            target_types='category',
            download=False,
            transform=test_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def visualize(self, img):
        plt.figure(figsize=(4,4))
        # CxHxW -> HxWxC
        img = img.permute(1, 2, 0)  
        # reverse normalization
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        # range [0, 1]
        img = torch.clamp(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.close()


# o = OxfordPetsDataset()
# train, test = o.get_dataloaders()
# print(len(train), "\n", len(test))
# i1, i2 = train[0], test[0]
# print(i1[0].shape, "\n", i2[0].shape)
# o.visualize(i1[0])