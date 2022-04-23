# 3D U-Net for Brain Segmentation in MR Images
This is the sourcecode for EGBE601 Class Project\
by Tanadul Somboonwong 6437981\
Reference from
> Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.\
> \
> Hwang, H., Rehman, H. Z. U., & Lee, S. (2019). 3D U-Net for skull stripping in brain MRI. Applied Sciences, 9(3), 569.

In this class project, preliminary results are covered. To determine the best model, various of model are investigated over 10 epochs by varying the subvoxel batch size. Therefore, the obtained best model is using subvoxel batch size = 1 without data augmentation and found that the highest scores are presented at epoch 7. The accuracy is 99.59%; F1 score is 98.04%. The segmentation results are very satisfied.


## Dataset Preparation
1. Download [NFBS Repository](http://preprocessed-connectomes-project.org/NFB_skullstripped/).
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
2. Adjust `SUBVOXEL_BATCH_SIZE`.
3. Run `inference.py`.
4. The program will take the images in `test_images` folder to process a segmentation.
5. The results will saved at `test_prediction_images` folder.

## Image Visualization
1. Open `image_visualization.py` or import it as a module others `*.py` file.
2. This module provides 2 visualization function:
- `visualize_image(image_path: str)`: visualize the image with slider bar to investigate another slices smoothly.
- `subplots_visualize_image(image_path: str, slice_list: list)`: visualize the image in subplots; selected the slices plane via adjust `slice_list`.
