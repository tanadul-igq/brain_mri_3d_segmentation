# 3D U-Net for Brain Segmentation in MR Images
This is the sourcecode for EGBE601 Class Project
*By Tanadul Somboonwong 6437981*

## Dataset Preparation
1. Download NFBS Repository
2. Extract the dataset into a project directory. Name the folder as `NFBS_Dataset`
3. Open `nfbs_tvt_split.py`
4. Run `nfbs_tvt_split.py` to split and reorganize `NFBS_Dataset` to produce `NFBS_Dataset_TVT`

## Model Training and Validation
1. Open `train.py`
2. Adjust the hyperparameters to obtain various of models. Adjustable hyperparameters consists of 
    `NUM_EPOCHS`: number of epochs
    `SUBVOXEL_BATCH_SIZE`: batch size of subvoxel
    `DATA_AUGMENTATION`: data augmentation trigger
    `LEARNING_RATE`: learning rate of a model
3. Run `train.py` to train a model
