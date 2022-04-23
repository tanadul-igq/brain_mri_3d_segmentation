import os
from glob import glob
import sys

import numpy
import nibabel
import torch
import torch.utils.data as td
from monai.data import DataLoader

from patchify import patchify
from tqdm.auto import tqdm
from pprint import pprint

from model import UNet
from utilities import unflatten_and_unpatch
from image_visualization import visualize_image, subplots_visualize_image

# hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 64
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
PATCH_STEP_SIZE = PATCH_SIZE
SUBVOXEL_BATCH_SIZE = 8

# global variables
TQDM_BARFORMAT = '{desc:28} {percentage:3.0f}%|{bar:50}{r_bar}'
TQDM_DISABLE = False
TQDM_KWARGS = {'leave': False, 'bar_format': TQDM_BARFORMAT, 'disable': TQDM_DISABLE}

def load_model(model, path_to_load):
    if os.path.isfile(path_to_load):
        checkpoint = torch.load(path_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'\nUnable to load checkpoint at {path_to_load}')
    return model

def patch_and_flatten(image):
    patched_image = patchify(image=image, patch_size=PATCH_SHAPE, step=PATCH_STEP_SIZE) # numpy(256, 256, 192) to numpy(4, 4, 3, 64, 64, 64)
    patched_image = torch.tensor(numpy.reshape(patched_image, (-1, patched_image.shape[-3], patched_image.shape[-2], patched_image.shape[-1]))) # numpy(4, 4, 3, 64, 64, 64) to tensor([48, 64, 64, 64])
    return patched_image

def save_as_nii(array, saved_name):
    nifti1_image = nibabel.Nifti1Image(dataobj=array, affine=numpy.eye(4))
    nibabel.save(nifti1_image, os.path.join('test_prediction_images', saved_name))

def segment_brain(model, image_path):
    
    # import nii image
    test_image = nibabel.load(filename=image_path).get_fdata()
    
    # patch and reshape
    patched_image = patch_and_flatten(image=test_image) # torch.Size([48, 64, 64, 64])
    
    # use dataloader to enable dataset batching
    test_subvoxel_dataset = td.TensorDataset(patched_image)
    test_subvoxel_loader = DataLoader(test_subvoxel_dataset, batch_size=SUBVOXEL_BATCH_SIZE)

    # initialize a tensor to store a concatenated prediction tensor for reconstruction image
    concat_prediction = torch.tensor([]).to(device=DEVICE)
    
    # inference
    model.eval()
    with torch.inference_mode():
        # feed the subvoxel into model
        subvoxel_loop = tqdm(test_subvoxel_loader, desc='Brain Segmentation', **TQDM_KWARGS)
        for subvoxel_batch_index, test_subvoxel in enumerate(subvoxel_loop):
            
            # prepare input data
            test_subvoxel = test_subvoxel[0].float().unsqueeze(1).to(device=DEVICE) # tensor([subvoxel_batch_size, 64, 64, 64]) to tensor([subvoxel_batch_size, 1, 64, 64, 64])
            
            # feed input to model
            patched_prediction = torch.sigmoid(model(test_subvoxel)) # tensor([subvoxel_batch_size, 1, 64, 64, 64])
            patched_prediction = (patched_prediction > 0.5).float() # tensor([subvoxel_batch_size, 1, 64, 64, 64]); this tensor is converted to binary (0., 1.)

            # concatenate the subvoxels of patched prediction for reconstruct the whole volume
            concat_prediction = torch.cat([concat_prediction, patched_prediction], dim=0) # concatenate tensor([subvoxel_batch_size, 1, 64, 64, 64]) then acheive tensor([48, 1, 64, 64, 64]) at last iteration

        # reconstruct subvoxel into whole volume 
        reconstructed_prediction = unflatten_and_unpatch(concat_prediction=concat_prediction)
        
        # save prediction image
        subject_name = image_path.split(sep='\\')[-1].split(sep='.')[0]
        prediction_saved_name = f'{subject_name}_pred_brainmask.nii.gz'
        save_as_nii(array=reconstructed_prediction, saved_name=prediction_saved_name)

if __name__ == '__main__':
    
    # input: model path and image path
    model_path = r'sbs1\checkpoints\sbs1_best_f1_score.pth.tar'
    test_images_path = glob(os.path.join('NFBS_Dataset_TVT', 'test_images', '*.nii.gz'))
    
    # model: U-Net with out_channels=1 meaning that the model is single class segmentation
    model = UNet(in_channels=1, out_channels=1).to(device=DEVICE)

    # load trained model
    model = load_model(model=model, path_to_load=model_path)
    
    # inference loop
    test_loop = tqdm(test_images_path, desc='Progress', **TQDM_KWARGS)
    for test_image_path in test_loop:
    
        # inference
        segment_brain(model=model, image_path=test_image_path)
        
    print('Brain Segmentation Done!')