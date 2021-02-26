"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import h5py
def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0



def get_mnist(data_dir='./data/mnist/', batch_size=128, resize= False):
    from torchvision.datasets import MNIST
    if resize:

        transform =  transforms.Compose([transforms.Resize((32)),transforms.ToTensor()])
        train = MNIST(root=data_dir, train=True, download=True,transform=transform)
        test = MNIST(root=data_dir, train=False, download=True, transform=transform)


        dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=4)

        return dataloader, test_dataloader
    else:
        transform = None
        train = MNIST(root=data_dir, train=True, download=True,target_transform=transform)
        test = MNIST(root=data_dir, train=False, download=True, target_transform=transform)

        #X = torch.cat([train.data.float().view(-1, 784) / 255., test.data.float().view(-1, 784) / 255.], 0)
        X = torch.cat([train.data.unsqueeze(1).float() / 255., test.data.unsqueeze(1).float() / 255.], 0)
        Y = torch.cat([train.targets, test.targets], 0)

        dataset = dict()
        dataset['X'] = X
        dataset['Y'] = Y

        dataloader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True, num_workers=4)

        return dataloader, dataset

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        #img = torch.from_numpy(img).float()
        return img , 0
class CustomImageFolderRandom(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

class CustomTensorDatasetRandom(Dataset):
    def __init__(self, data_tensor, transform=None):
        (self.data_tensor,self.labels) = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img1 = torch.from_numpy(img1).float()
        img2 = self.data_tensor[index2]
        img2 = torch.from_numpy(img2).float()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = self.labels[index1]
        #label = torch.from_numpy(label).float()labels

        
        return img1, img2 , label

    def __len__(self):
        return self.data_tensor.shape[0]

class Dataset3DChairsRAM(Dataset):
    def __init__(self,images):
        self.data_tensor = images
    def __getitem__(self, index1):

        img1 = self.data_tensor[index1]
        #img1 = torch.from_numpy(img1).float()

        return img1, 0

    def __len__(self):
        return self.data_tensor.shape[0]



class Fullloaded_dataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        #print("reached point 2")
        (self.data_tensor,self.labels) = data_tensor
        
        self.transform = transform
        self.indices = range(len(self))
        
    def __getitem__(self, index1):

        img1 = self.data_tensor[index1]
        img1 = torch.from_numpy(img1).float()
        label = self.labels[index1]
        #label = torch.from_numpy(label).float()labels
        if img1.shape[1] == 3:
            img1 /= 255
        if self.transform is not None:
            img1 = self.transform(img1)
        

        return img1, label

    def __len__(self):
        return self.data_tensor.shape[0]


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    #assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])
    if name.lower() == 'mnist':
        if args.model == 'jointvae':
            DL,dset = get_mnist(data_dir='./data', batch_size=args.batch_size, resize=True)
        else:
            DL,dset = get_mnist(data_dir='./data', batch_size=args.batch_size)

        return DL
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == '3dchairs':
        
        root = os.path.join(dset_dir, '3DChairs')
        if not os.path.exists(os.path.join(root,'3dchairs.pt')):
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),])

            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder(**train_kwargs)
            
            dl = DataLoader(dset, batch_size=86366)
            dl = iter(dl)
            images, _ = dl.next()
            torch.save(images,os.path.join(root,'3dchairs.pt'))
        else:
            images = torch.load(os.path.join(root,'3dchairs.pt'))
        train_kwargs = {'images':images}
        dset = Dataset3DChairsRAM
    elif name.lower() == 'dsprites':
        #print("reached point one")
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        #print(data.shape)
        imgs = np.expand_dims(data['imgs'], axis=1)
        labels = data['latents_values'][:,1]
        
        #data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':(imgs,labels)}
        if args.factorvae_tc:
            dset = CustomTensorDatasetRandom
        else: 
            dset =  Fullloaded_dataset
    elif name.lower() == '3dshapes':
        dataset = h5py.File(os.path.join(dset_dir,'3Dshapes/3dshapes.h5'),'r')
        
        imgs = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        imgs = np.asarray(imgs)
        imgs = np.transpose(imgs,(0,3,2,1))
        
        print(imgs.shape)
        labels = dataset['labels']  # array shape [480000,6], float64
        train_kwargs = {'data_tensor':(imgs,labels)}
        if args.factorvae_tc:
            dset = CustomTensorDatasetRandom
        else: 
            dset =  Fullloaded_dataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers,
                              #pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

