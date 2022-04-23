import os
from glob import glob

import monai.transforms as mt
from monai.data import Dataset, DataLoader

def get_subject_name(dataset_dir, image_folder_dir, batch_index):
    image_dir = os.path.join(dataset_dir, image_folder_dir)
    subject_list = os.listdir(image_dir)
    subject_name = subject_list[batch_index].split(sep='.')[0]
    return subject_name

def generate_dataset_dict(dataset_dir, image_folder_dir, mask_folder_dir):
    images_dir_list = glob(os.path.join(dataset_dir, image_folder_dir, '*.nii.gz'))
    masks_dir_list = glob(os.path.join(dataset_dir, mask_folder_dir, '*.nii.gz'))
    dataset_dict = [{'image': image_dir, 'mask': mask_dir} for image_dir, mask_dir in zip(images_dir_list, masks_dir_list)]
    return dataset_dict
    
def generate_dataloader(dataset_dir, train_image_folder_dir, train_mask_folder_dir, val_image_folder_dir, val_mask_folder_dir, data_augmentation, dataset_batch_size=1):
    
    # dataset dictionary
    train_dataset_dict = generate_dataset_dict(dataset_dir=dataset_dir, image_folder_dir=train_image_folder_dir, mask_folder_dir=train_mask_folder_dir)
    val_dataset_dict = generate_dataset_dict(dataset_dir=dataset_dir, image_folder_dir=val_image_folder_dir, mask_folder_dir=val_mask_folder_dir)
    
    # data augmentation
    if data_augmentation:
        train_transforms = mt.Compose(
            [
                mt.LoadImaged(keys=['image', 'mask']),
                mt.AddChanneld(keys=['image', 'mask']),
                mt.RandGaussianSmoothd(keys='image', prob=0.3, sigma_x=(1, 2)),
                mt.RandGibbsNoised(keys='image', prob=0.3, alpha=(0.6, 0.8)),
                mt.RandKSpaceSpikeNoised(keys='image', prob=0.3, intensity_range=(10, 13)),
                mt.ToTensord(keys=['image', 'mask']),
            ]
        )
        # default augmentation
        val_transforms = mt.Compose(
            [
                mt.LoadImaged(keys=['image', 'mask']),
                mt.AddChanneld(keys=['image', 'mask']),
                mt.ToTensord(keys=['image', 'mask'])
            ]
        )
    else:
        # default augmentation
        train_transforms = mt.Compose(
            [
                mt.LoadImaged(keys=['image', 'mask']),
                mt.AddChanneld(keys=['image', 'mask']),
                mt.ToTensord(keys=['image', 'mask']),
            ]
        )
        val_transforms = mt.Compose(
            [
                mt.LoadImaged(keys=['image', 'mask']),
                mt.AddChanneld(keys=['image', 'mask']),
                mt.ToTensord(keys=['image', 'mask'])
            ]
        )
  
    # dataloader
    train_dataset = Dataset(data=train_dataset_dict, transform=train_transforms)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=dataset_batch_size, shuffle=False)    
    val_dataset = Dataset(data=val_dataset_dict, transform=val_transforms)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=dataset_batch_size, shuffle=False)    
    
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    pass