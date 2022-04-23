import os

import numpy
import nibabel
import pandas
import torch

from patchify import patchify, unpatchify

# hyperparameters
PATCH_SIZE = 64
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
PATCH_STEP_SIZE = PATCH_SIZE

def save_checkpoint(epoch, model, optimizer, loss, directory, filename='model_checkpoint'):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    path_to_save = os.path.join(directory, filename + '.pth.tar')
    torch.save(checkpoint, path_to_save)
    print(f'\nCheckpoint is saved at {path_to_save}')
    print(f'Model and Optimizer are saved at Epoch {epoch}')
    
def load_checkpoint(model, optimizer, directory, filename='model_checkpoint'):
    path_to_load = os.path.join(directory, filename + '.pth.tar')
    if os.path.isfile(path_to_load):
        print(f'\nLoading checkpoint at {path_to_load}')
        checkpoint = torch.load(path_to_load)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f'Model and Optimizer are loaded and continued at Epoch {start_epoch}')
    else:
        print(f'\nUnable to load checkpoint at {path_to_load}')
    return start_epoch, model, optimizer, loss

def patch_and_flatten(image, mask):
    patched_image = patchify(image=image, patch_size=PATCH_SHAPE, step=PATCH_STEP_SIZE) # numpy(256, 256, 192) to numpy(4, 4, 3, 64, 64, 64)
    patched_mask = patchify(image=mask, patch_size=PATCH_SHAPE, step=PATCH_STEP_SIZE) # numpy(256, 256, 192) to numpy(4, 4, 3, 64, 64, 64)
    patched_image = torch.tensor(numpy.reshape(patched_image, (-1, patched_image.shape[-3], patched_image.shape[-2], patched_image.shape[-1]))) # numpy(4, 4, 3, 64, 64, 64) to tensor([48, 64, 64, 64])
    patched_mask =  torch.tensor(numpy.reshape(patched_mask, (-1, patched_mask.shape[-3], patched_mask.shape[-2], patched_mask.shape[-1]))) # # numpy(4, 4, 3, 64, 64, 64) to tensor([48, 64, 64, 64])
    return patched_image, patched_mask

def unflatten_and_unpatch(concat_prediction):
    concat_prediction = concat_prediction.squeeze(1).cpu().numpy() # tensor([48, 1, 64, 64, 64]) to numpy(48, 64, 64, 64)
    unflatten_prediction = numpy.reshape(concat_prediction, (4, 4, 3, 64, 64, 64)) # numpy(48, 64, 64, 64) to numpy(4, 4, 3, 64, 64, 64)
    reconstructed_prediction = unpatchify(unflatten_prediction, (256, 256, 192)) # numpy(4, 4, 3, 64, 64, 64) to numpy(256, 256, 192)
    return reconstructed_prediction

def scoring(groundtruth, prediction):
    # groundtruth (y) and prediction (y^) are both numpy array with binary values (0., 1.)
    
    # True Positive (TP): prediction is 1 (positive), and groundtruth is 1 (positive).
    TP = numpy.sum(numpy.logical_and(prediction == 1, groundtruth == 1))

    # True Negative (TN): prediction is 0 (negative), and groundtruth is 0 (negative).
    TN = numpy.sum(numpy.logical_and(prediction == 0, groundtruth == 0))

    # False Positive (FP): prediction is 1 (positive), but groundtruth is 0 (negative).
    FP = numpy.sum(numpy.logical_and(prediction == 1, groundtruth == 0))

    # False Negative (FN): prediction is 0 (negative), but groundtruth is 1 (positive).
    FN = numpy.sum(numpy.logical_and(prediction == 0, groundtruth == 1))
    
    # accuracy: 1 means ideally perfect prediction
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # precision: 1 means ideally perfect prediction
    precision = TP / (TP + FP)
    
    # sensitivity (aka recall, true positive rate)
    sensitivity = TP / (TP + FN)
    
    # specificity (aka selectivity, true negative rate)
    specificity = TN / (TN + FP)
    
    # F1 score (aka dice coefficient): 1 means ideally perfect prediction
    f1_score = (2 * TP) / ((2 * TP) + FP + FN)
    
    # DataFrame columns
    scores_dict = {
        'TP': [TP],
        'TN': [TN],
        'FP': [FP],
        'FN': [FN],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'F1 Score': [f1_score]
    }
    scores = pandas.DataFrame(scores_dict)
    
    return scores

def save_as_nii(array, saved_name):
    nifti1_image = nibabel.Nifti1Image(dataobj=array, affine=numpy.eye(4))
    nibabel.save(nifti1_image, os.path.join('prediction_images', saved_name + '.nii.gz'))

if __name__ == '__main__':
    pass