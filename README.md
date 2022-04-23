# 3D U-Net for Brain Segmentation in MR Images
This is the sourcecode for EGBE601 Class Project
*By Tanadul Somboonwong 6437981*

## Dataset Preparation
1. Download NFBS Repository.
2. Extract the dataset into a project directory. Name the folder as `NFBS_Dataset`.
4. Open `nfbs_tvt_split.py`.
5. Run `nfbs_tvt_split.py` to split and reorganize `NFBS_Dataset` to produce `NFBS_Dataset_TVT`.

## Model Training and Validation
1. Open `train.py`.
2. Adjust the hyperparameters to obtain various of models. Adjustable hyperparameters consists of 
- `NUM_EPOCHS`: number of epochs
- `SUBVOXEL_BATCH_SIZE`: batch size of subvoxel
- `DATA_AUGMENTATION`: data augmentation trigger
- `LEARNING_RATE`: learning rate of a model
3. Run `train.py` to train a model.
4. After complete 1 epoch, the model checkpoint, prediction results, and score statistics will be saved in the model folder.
5. Therefore, after complete `NUM_EPOCHS` epochs, the model folder will consist of checkpoints, prediction results, and score statistics of every epochs.

## Inference
1. Open `inference.py`.
2. Adjust `SUBVOXEL_BATCH_SIZE`
3. Run `inference.py`.
4. The results will saved at `test_prediction_images` folder.

## Image Visualization
1. Open `image_visualization.py` or import it as a module others `*.py` file.
2. This module provides 2 visualization function
- `visualize_image(image_path: str)`: visualize the image with slider bar to investigate another slices smoothly.
- `subplots_visualize_image(image_path: str, slice_list: list)`: visualize the image in subplots; selected the slices plane via adjust `slice_list`.
