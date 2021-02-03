from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FaceScrub(Dataset):
    def __init__(self, path, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(path)
        #actor_images = input['actor_images']
        #actor_labels = input['actor_labels']
        #actress_images = input['actress_images']
        #actress_labels = input['actress_labels']

        #data = np.concatenate([actor_images, actress_images], axis=0)
        #labels = np.concatenate([actor_labels, actress_labels], axis=0)
        data = input['images']/255.0
        labels = input['labels']

        '''
        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        '''
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FaceScrub_out(Dataset):
    def __init__(self, path, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(path)

        data = np.squeeze(input['images'])
        out = input['out']

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        out = out[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.out = out[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.out = out[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, out = self.data[index], self.out[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, out


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(self.root)
        data = input/255.0

        '''
        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        '''
        labels = np.array([0] * data.shape[0])
        
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CelebA_out(Dataset):
    def __init__(self, root, transform=None, target_transform=None, num = 50000):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(self.root)
        data = np.squeeze(input['images'])
        out = input['out']

        self.data = data[:num]
        #self.labels = labels
        self.out = out[:num]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img,out = self.data[index], self.out[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, out

