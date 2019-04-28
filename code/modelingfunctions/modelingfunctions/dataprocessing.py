import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Sampler
import torch
import pandas as pd
import random


# Setting up paths. This assums clean_data is one directory up from the
# path Jupyter was started in
base_path = Path('../clean_data/').absolute()

# Most basic data. All targets
raw_base_path = base_path / 'motorcycles'
raw_train_path = str(raw_base_path / 'train')
raw_val_path = str(raw_base_path / 'val')
raw_test_path = str(raw_base_path / 'test')

# Data made square. All targets
square_base_path = base_path / 'square_motorcycles'
square_train_path = str(square_base_path / 'train')
square_val_path = str(square_base_path / 'val')
square_test_path = str(square_base_path / 'test')

# Data with at least eight images
raw_base_path8 = base_path / 'eight_min'
raw_train_path8 = str(raw_base_path8 / 'train')
raw_val_path8 = str(raw_base_path8 / 'val')
raw_test_path8 = str(raw_base_path8 / 'test')

# Data with at least eight images with val=0.3 and test=0.3
raw_base_path8_balanced = base_path / 'eight_min_balanced'
raw_train_path8_balanced = str(raw_base_path8_balanced / 'train')
raw_val_path_8_balanced = str(raw_base_path8_balanced / 'val')
raw_test_path_8_balanced = str(raw_base_path8_balanced / 'test')

# Data with at least eight images with val=0.1 and test=0.1
raw_base_path8_nonbalanced = base_path / 'eight_min_nonbalanced'
raw_train_path8_nonbalanced = str(raw_base_path8_nonbalanced / 'train')
raw_val_path8_nonbalanced = str(raw_base_path8_nonbalanced / 'val')
raw_test_path8_nonbalanced = str(raw_base_path8_nonbalanced / 'test')

# Data with fewer than eight images with val=0.2 and test=0.2
raw_base_path7 = base_path / 'seven_max'
raw_train_path7 = str(raw_base_path7 / 'train')
raw_val_path7 = str(raw_base_path7 / 'val')
raw_test_path7 = str(raw_base_path7 / 'test')


'''******************* Transforms *****************************'''
# Most basic, centercrop
basic_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=244),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=244),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Basic transforms with smaller images
basic_transforms_small = {
    'train': transforms.Compose([
        transforms.Resize(size=192),
        transforms.CenterCrop(size=183),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=192),
        transforms.CenterCrop(size=183),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# Basic transform but capturing wide image (not losing handlebars and back)
basic_transforms_wide = {
    'train': transforms.Compose([
        transforms.Resize(size=(256, 192)),
        transforms.CenterCrop(size=(256, 192)),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(256, 192)),
        transforms.CenterCrop(size=(256, 192)),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

basic_transforms_wide_large = {
    'train': transforms.Compose([
        transforms.Resize(size=(384, 288)),
        transforms.CenterCrop(size=(384, 288)),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(384, 288)),
        transforms.CenterCrop(size=(384, 288)),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Transforms with random data augmentation
complex_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Transforms with random data augmentation and wide format
complex_transforms_wide = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 192), scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=(256, 192)),
        transforms.ToTensor(),
        # Normalize using same mean, std as imagenet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(256, 192)),
        transforms.CenterCrop(size=(256, 192)),
        transforms.ToTensor(),
        # Normalize using same mean, std as ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

basic_data = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=basic_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=basic_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=basic_transforms['valid'])
}

basic_data_small = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=basic_transforms_small['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=basic_transforms_small['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=basic_transforms_small['valid'])
}

basic_data_wide = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=basic_transforms_wide['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=basic_transforms_wide['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=basic_transforms_wide['valid'])
}

basic_data_wide_large = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=basic_transforms_wide_large['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=basic_transforms_wide_large['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=basic_transforms_wide_large['valid'])
}


complex_data = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=complex_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=complex_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=complex_transforms['valid'])
}

complex_data_wide = {
    'train': datasets.ImageFolder(root=raw_train_path,
                                  transform=complex_transforms_wide['train']),
    'valid': datasets.ImageFolder(root=raw_val_path,
                                  transform=complex_transforms_wide['valid']),
    'test': datasets.ImageFolder(root=raw_test_path,
                                 transform=complex_transforms_wide['valid'])
}

complex_data8 = {
    'train': datasets.ImageFolder(root=raw_train_path8,
                                  transform=complex_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path8,
                                  transform=complex_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path8,
                                 transform=complex_transforms['valid'])
}

complex_data8_balanced = {
    'train': datasets.ImageFolder(root=raw_train_path8_balanced,
                                  transform=complex_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path_8_balanced,
                                  transform=complex_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path_8_balanced,
                                 transform=complex_transforms['valid'])
}

complex_data8_nonbalanced = {
    'train': datasets.ImageFolder(root=raw_train_path8_nonbalanced,
                                  transform=complex_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path8_nonbalanced,
                                  transform=complex_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path8_nonbalanced,
                                 transform=complex_transforms['valid'])
}

complex_data7 = {
    'train': datasets.ImageFolder(root=raw_train_path7,
                                  transform=complex_transforms['train']),
    'valid': datasets.ImageFolder(root=raw_val_path7,
                                  transform=complex_transforms['valid']),
    'test': datasets.ImageFolder(root=raw_test_path7,
                                 transform=complex_transforms['valid'])
}

''' ********************** Data loaders ************************** '''


class TargetSampler(Sampler):
    '''
    Sampler base on the number of targets we want. Rather than random sampling
    from all targets, which may not have the same targets as test and train,
    we pull all images from the first x number of targets

    This was extremely slow at first, because I was iterating through the data
    instead of just the targets. Now it takes no noticable time

    '''

    def __init__(self, data, num_targets):
        '''
        data: A pytorch dataset (In this case, an ImageFolder)
        num_targets: Int representing the first X targets to use.
        '''
        self.num_targets = num_targets
        self.data = data

    def __iter__(self):
        indices = []
        for index, target in enumerate(self.data.targets):
            # Add the indice if the target is in range
            if target < self.num_targets:
                indices.append(index)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # Count the number of images in the target range
        # The image is a tuple, with image[1] being the target
        return len([i for i in self.data.targets if i < self.num_targets])


def get_target_dataloader(data, num_targets, batch_size, pin_memory=False,
                          num_workers=0):
    '''
    Creates a dataloader with a limited number of targets
    data: A Pytorch DataSet with train, valid and test data
    '''
    train_sampler = TargetSampler(data['train'], num_targets=num_targets)
    val_sampler = TargetSampler(data['valid'], num_targets=num_targets)
    test_sampler = TargetSampler(data['test'], num_targets=num_targets)

    return {
        'train': DataLoader(data['train'], batch_size=batch_size,
                            sampler=train_sampler, shuffle=False,
                            pin_memory=pin_memory, num_workers=num_workers),
        'val': DataLoader(data['valid'], batch_size=batch_size,
                          sampler=val_sampler, shuffle=False,
                          pin_memory=pin_memory, num_workers=num_workers),
        'test': DataLoader(data['test'], batch_size=batch_size,
                           sampler=test_sampler, shuffle=False,
                           pin_memory=pin_memory, num_workers=num_workers)
    }
