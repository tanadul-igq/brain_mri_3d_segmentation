import os
import shutil
import numpy

def nfbs_tvt_split():

    # path and constant
    nfbs_dataset_source_path = 'NFBS_Dataset'
    nfbs_dataset_destination_path = 'NFBS_Dataset_TVT'
    os.makedirs(nfbs_dataset_destination_path)
    subject_list = sorted(os.listdir(nfbs_dataset_source_path))
    dataset_size = len(subject_list)
    
    # define portion as train:validation:test = 80%:16%:4%
    train_portion = 0.8
    val_test_portion = round(1 - train_portion, 2)
    test_portion = round(0.2 * val_test_portion, 2)
    val_portion = val_test_portion - test_portion
    
    # allocate train dataset
    train_size = int(numpy.ceil(train_portion * dataset_size))
    train_start_index = 0
    train_last_index = train_start_index + train_size
    train_subject_list = subject_list[train_start_index : train_last_index]
    
    # allocate validation dataset
    val_size = int(numpy.ceil(val_portion * dataset_size))
    val_start_index = train_last_index
    val_last_index = val_start_index + val_size   
    val_subject_list = subject_list[val_start_index : val_last_index]

    # allocate validation dataset
    test_size = int(numpy.floor(test_portion * dataset_size))
    test_start_index = val_last_index
    test_last_index = test_start_index + test_size   
    test_subject_list = subject_list[test_start_index : test_last_index]
    
    for subject in subject_list:
        
        # check train / validation, and update the folder string
        if subject in train_subject_list:
            images_destination_folder = 'train_images'
            masks_destination_folder = 'train_masks'
            
        elif subject in val_subject_list:
            images_destination_folder = 'val_images'
            masks_destination_folder = 'val_masks'
            
        elif subject in test_subject_list:
            images_destination_folder = 'test_images'
            masks_destination_folder = 'test_masks'

        # extract and generate path string
        subject_path = os.path.join(nfbs_dataset_source_path, subject)
        files_in_subject_path = os.listdir(subject_path)
        image_path = os.path.join(subject_path, files_in_subject_path[0])
        mask_path = os.path.join(subject_path, files_in_subject_path[2])
        image_destination_path = os.path.join(nfbs_dataset_destination_path, images_destination_folder)
        mask_destination_path = os.path.join(nfbs_dataset_destination_path, masks_destination_folder)

        # in case of original image
        if not os.path.exists(image_destination_path):
            os.makedirs(image_destination_path)
        shutil.copy(src=image_path, dst=image_destination_path)

        # in case of brain mask
        if not os.path.exists(mask_destination_path):
            os.makedirs(mask_destination_path)
        shutil.copy(src=mask_path, dst=mask_destination_path)
            
    print('Rearrange NFBS Dataset Successfully!')
 
if __name__ == '__main__':
    nfbs_tvt_split()