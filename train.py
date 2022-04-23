import os
from distutils.dir_util import copy_tree

import numpy
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
from monai.data import DataLoader

from tqdm.auto import tqdm

from dataset import get_subject_name, generate_dataloader
from model import UNet
from utilities import save_checkpoint, load_checkpoint, patch_and_flatten, unflatten_and_unpatch, scoring, save_as_nii

# hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = 'NFBS_Dataset_TVT'
TRAIN_IMAGE_FOLDER = 'train_images'
TRAIN_MASK_FOLDER = 'train_masks' 
VAL_IMAGE_FOLDER = 'val_images' 
VAL_MASK_FOLDER = 'val_masks'
NUM_EPOCHS = 10
DATASET_BATCH_SIZE = 1 # force to be 1; mean that only 1 subject will be loaded at a time
SUBVOXEL_BATCH_SIZE = 4
DATA_AUGMENTATION = False
LEARNING_RATE = 1e-4

# global control variables
LOAD_CHECKPOINT = False
aug_suffix = '_aug' if DATA_AUGMENTATION else ''
model_trial_name = f'sbs{SUBVOXEL_BATCH_SIZE}{aug_suffix}'
model_to_load = f'{model_trial_name}_epoch2'
checkpoints_folder = f'{model_trial_name}/checkpoints'
results_folder = f'{model_trial_name}/results'
statistics_folder = f'{model_trial_name}/statistics'
scores_file_dir = f'{statistics_folder}/{model_trial_name}_scores.csv'
loss_file_dir = f'{statistics_folder}/{model_trial_name}_loss.csv'
best_accuracy = 0
best_f1_score = 0
epochs_scores = pandas.DataFrame()
epochs_loss = numpy.array([])
TQDM_BARFORMAT = '{desc:28} {percentage:3.0f}%|{bar:50}{r_bar}'
TQDM_DISABLE = False
TQDM_KWARGS = {'leave': False, 'bar_format': TQDM_BARFORMAT, 'disable': TQDM_DISABLE}

def single_epoch_training(train_dataloader, model, optimizer, loss_fn, scaler, epoch):
    
    # initialize variables
    global epochs_loss
    
    # this loop access data of single subject which still have a size of (256, 256, 192)
    single_data_loop = tqdm(train_dataloader, position=1, **TQDM_KWARGS)
    for dataset_batch_index, data in enumerate(single_data_loop):
        
        # set progressbar description
        single_data_loop.set_description(f'Train Dataset Batch {dataset_batch_index}')
        
        # prepare tensor to patch
        image = data['image'].squeeze(0).squeeze(0).numpy() # tensor([1, 1, 256, 256, 192]) to numpy(256, 256, 192)
        mask = data['mask'].squeeze(0).squeeze(0).numpy() # tensor([1, 1, 256, 256, 192]) to numpy(256, 256, 192)

        # patch and prepare tensor
        patched_image, patched_mask = patch_and_flatten(image=image, mask=mask)
        
        # construct patched data as TensorDataset and DataLoader
        train_subvoxel_dataset = td.TensorDataset(patched_image, patched_mask) # tensor([48, 64, 64, 64]), tensor([48, 64, 64, 64])
        train_subvoxel_loader = DataLoader(train_subvoxel_dataset, batch_size=SUBVOXEL_BATCH_SIZE)
        
        # this loop access data of single subject in term of 48 subvoxels obtained from patching
        subvoxels_loop = tqdm(train_subvoxel_loader, position=2, **TQDM_KWARGS)
        for subvoxel_batch_index, (patched_image, patched_mask) in enumerate(subvoxels_loop):
            
            # set progressbar description
            subvoxels_loop.set_description(f'Subvoxels Batch {subvoxel_batch_index}')
            
            # prepare data: add channel dimension to data
            patched_image = patched_image.float().unsqueeze(1).to(device=DEVICE) # tensor([subvoxel_batch_size, 64, 64, 64]) to tensor([subvoxel_batch_size, 1, 64, 64, 64])
            patched_mask = patched_mask.float().unsqueeze(1).to(device=DEVICE) # tensor([subvoxel_batch_size, 64, 64, 64]) to tensor([subvoxel_batch_size, 1, 64, 64, 64])
            
            # forward with float16 system
            with torch.cuda.amp.autocast():        
                optimizer.zero_grad()
                patched_prediction = model(patched_image) # tensor([subvoxel_batch_size, 1, 64, 64, 64])
            
            # backward
            loss = loss_fn(patched_prediction, patched_mask)
            scaler.scale(loss).backward()
            
            # gradient descent
            scaler.step(optimizer)
            scaler.update()
        
        # set progressbar postfix
        single_data_loop.set_postfix(loss=loss.item())
    
    # save loss of an epoch
    epochs_loss = numpy.append(epochs_loss, loss.item())
    numpy.savetxt(loss_file_dir, epochs_loss, delimiter=',')
    
    # save trained model
    save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, loss=loss_fn, directory=checkpoints_folder, filename=f'{model_trial_name}_epoch{epoch}')

def single_epoch_validation(val_dataloader, model, optimizer, loss_fn, epoch):
    
    # initialize variables
    global epochs_scores, best_accuracy, best_f1_score
    single_dataset_score_history = pandas.DataFrame()
    
    model.eval()
    with torch.no_grad():
    
        # this loop access data of single subject which still have a size of (256, 256, 192)
        single_data_loop = tqdm(val_dataloader, position=1, **TQDM_KWARGS)
        for dataset_batch_index, data in enumerate(single_data_loop):
            
            # initialize a tensor to store a concatenated prediction tensor for reconstruction image
            concat_prediction = torch.tensor([]).to(device=DEVICE)
            
            # set progressbar description
            single_data_loop.set_description(f'Validation Dataset Batch {dataset_batch_index}')
            
            # prepare tensor to patch
            image = data['image'].squeeze(0).squeeze(0).numpy() # tensor([1, 1, 256, 256, 192]) to numpy(256, 256, 192)
            mask = data['mask'].squeeze(0).squeeze(0).numpy() # tensor([1, 1, 256, 256, 192]) to numpy(256, 256, 192)

            # patch and prepare tensor
            patched_image, patched_mask = patch_and_flatten(image=image, mask=mask)
            
            # construct patched data as TensorDataset and DataLoader
            train_subvoxel_dataset = td.TensorDataset(patched_image, patched_mask) # tensor([48, 64, 64, 64]), tensor([48, 64, 64, 64])
            train_subvoxel_loader = DataLoader(train_subvoxel_dataset, batch_size=SUBVOXEL_BATCH_SIZE)
            
            # this loop access data of single subject in term of 48 subvoxels obtained from patching
            subvoxels_loop = tqdm(train_subvoxel_loader, position=2, **TQDM_KWARGS)
            for subvoxel_batch_index, (patched_image, patched_mask) in enumerate(subvoxels_loop):
                
                # set progressbar description
                subvoxels_loop.set_description(f'Subvoxels Batch {subvoxel_batch_index}')
               
                # prepare data: add channel dimension to data
                patched_image = patched_image.float().unsqueeze(1).to(device=DEVICE) # tensor([subvoxel_batch_size, 64, 64, 64]) to tensor([subvoxel_batch_size, 1, 64, 64, 64])
                patched_mask = patched_mask.float().unsqueeze(1).to(device=DEVICE) # tensor([subvoxel_batch_size, 64, 64, 64]) to tensor([subvoxel_batch_size, 1, 64, 64, 64])

                # validation
                patched_prediction = torch.sigmoid(model(patched_image)) # tensor([subvoxel_batch_size, 1, 64, 64, 64])
                patched_prediction = (patched_prediction > 0.5).float() # tensor([subvoxel_batch_size, 1, 64, 64, 64]); this tensor is converted to binary (0., 1.)
                
                # concatenate the subvoxels of patched prediction for reconstruct the whole volume
                concat_prediction = torch.cat([concat_prediction, patched_prediction], dim=0) # concatenate tensor([subvoxel_batch_size, 1, 64, 64, 64]) then acheive tensor([48, 1, 64, 64, 64]) at last iteration
                        
            # reconstruct subvoxel into whole volume 
            reconstructed_prediction = unflatten_and_unpatch(concat_prediction=concat_prediction)
            
            # save whole volume as *.nii.gz
            subject_name = get_subject_name(dataset_dir=DATASET_DIR, image_folder_dir=VAL_IMAGE_FOLDER, batch_index=dataset_batch_index)
            prediction_saved_name = f'{subject_name}_pred_brainmask'
            save_as_nii(array=reconstructed_prediction, saved_name=prediction_saved_name)
            
            # model scoring of each single dataset
            single_dataset_score = scoring(groundtruth=mask, prediction=reconstructed_prediction)
            single_dataset_score_history = pandas.concat([single_dataset_score_history, single_dataset_score], ignore_index=True)
    
    # scores of each epoch can be obtained from 
    # 1) the sum of confusion matrix value (TP, TN, FP, FN)
    # 2) the mean of metrics score (Accuracy, Precision, Sensitivity, Specificity, F1 Score)
    sum_confusion_values = pandas.DataFrame(single_dataset_score_history[['TP', 'TN', 'FP', 'FN']].sum()).T
    mean_metrics_scores = pandas.DataFrame(single_dataset_score_history[['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score']].mean()).T
    single_epoch_scores = pandas.concat([sum_confusion_values, mean_metrics_scores], axis=1, ignore_index=False)
    epochs_scores = pandas.concat([epochs_scores, single_epoch_scores], ignore_index=True)
    
    # save best model in term of best accuracy and f1 score; then backup the best prediction result
    latest_accuracy = epochs_scores['Accuracy'].iloc[-1]
    if latest_accuracy >= best_accuracy:
        best_accuracy = latest_accuracy
        save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, loss=loss_fn, directory=checkpoints_folder, filename=f'{model_trial_name}_best_accuracy')
        copy_tree(src='prediction_images', dst=f'{results_folder}/best_accuracy')
    
    latest_f1_score = epochs_scores['F1 Score'].iloc[-1]
    if latest_f1_score >= best_f1_score:
        best_f1_score = latest_f1_score
        save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, loss=loss_fn, directory=checkpoints_folder, filename=f'{model_trial_name}_best_f1_score')
        copy_tree(src='prediction_images', dst=f'{results_folder}/best_f1_score')
    
    # backup the prediction image for every epoch
    copy_tree(src='prediction_images', dst=f'{results_folder}/epoch_{epoch}')
        
    # save the score from every epoch as a history
    epochs_scores.to_csv(scores_file_dir)
    
def n_epochs_training():
    
    global epochs_scores, epochs_loss
    
    if os.path.exists(model_trial_name):
        if os.path.exists(scores_file_dir):
            epochs_scores = pandas.read_csv(scores_file_dir, index_col=0)
            epochs_loss = numpy.loadtxt(loss_file_dir, delimiter=',')
    else:
        # create folder to contain the files produced by the model
        os.makedirs(checkpoints_folder)
        os.makedirs(results_folder)
        os.makedirs(statistics_folder)
    
    # dataloader
    train_dataloader, val_dataloader = generate_dataloader(
        dataset_dir=DATASET_DIR, 
        train_image_folder_dir=TRAIN_IMAGE_FOLDER, 
        train_mask_folder_dir=TRAIN_MASK_FOLDER, 
        val_image_folder_dir=VAL_IMAGE_FOLDER, 
        val_mask_folder_dir=VAL_MASK_FOLDER,
        data_augmentation=DATA_AUGMENTATION, 
        dataset_batch_size=DATASET_BATCH_SIZE
    )

    # model: U-Net with out_channels=1 meaning that the model is single class segmentation
    model = UNet(in_channels=1, out_channels=1).to(device=DEVICE)
    
    # loss function: BCEWithLogitsLoss() suits with any number of classes in segmentation
    loss = nn.BCEWithLogitsLoss()
    
    # optimizer: Gradient Descent via Adam algorithm
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    # scaler: enable auto mixed precision training (amp) where it changes float32 into float16 data
    scaler = torch.cuda.amp.GradScaler()    
    
    # load the checkpoint: model and optimizer are required beforehand
    start_epoch = 0
    if LOAD_CHECKPOINT:
        start_epoch, model, optimizer, loss = load_checkpoint(model=model, optimizer=optimizer, directory=checkpoints_folder, filename=model_to_load)
    
    # training network
    # this loop enable iterations over the pipeline (train and validate) for NUM_EPOCHS times
    model.train()
    epochs_loop = tqdm(range(start_epoch, NUM_EPOCHS), position=0, **TQDM_KWARGS)
    for epoch in epochs_loop:
        
        # set progressbar description
        epochs_loop.set_description(f'Epoch {epoch}')
        
        # train 
        single_epoch_training(train_dataloader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss, scaler=scaler, epoch=epoch)

        # validate
        single_epoch_validation(val_dataloader=val_dataloader, model=model, optimizer=optimizer, loss_fn=loss, epoch=epoch)
        
if __name__ == '__main__':
    n_epochs_training()