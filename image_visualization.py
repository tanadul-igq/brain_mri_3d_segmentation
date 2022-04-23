import os
from glob import glob

import nibabel
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def visualize_image(image_path: str):
    
    # import nii image
    image_obj = nibabel.load(image_path)

    # extract the data as numpy array
    image_data = image_obj.get_fdata()
    image_height, image_width, image_depth = image_data.shape # (256, 256, 192)

    # ////////////////////////////////////////////////////////////////////////////////////////////

    # define initial parameters in visualization
    init_layer = int((image_depth - 1) / 2)

    # create the figure and show the image that we will manipulate
    fig, ax = plt.subplots()
    visualized_image = plt.imshow(image_data[:, :, init_layer], cmap='gray')
    # plt.suptitle(image_path.split(sep='\\')[-2])
    # plt.title(image_path.split(sep='\\')[-1])    
    
    # ////////////////////////////////////////////////////////////////////////////////////////////

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.10, bottom=0.25)

    # Make a horizontal slider to control the layer.
    layer_axis = plt.axes([0.25, 0.1, 0.50, 0.03])
    layer_slider = Slider(
        ax=layer_axis,
        label='Layer',
        valmin=0,
        valmax=image_depth - 1,
        valinit=init_layer,
        valstep=1,
    )

    # the function to be called anytime a slider's value changes
    def update(updated_layer):
        visualized_image.set_data(image_data[:, :, int(updated_layer)])
        fig.canvas.draw_idle()

    # register the update function into a layer slider
    layer_slider.on_changed(update)
    
    # ////////////////////////////////////////////////////////////////////////////////////////////

    # create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    reset_axis = plt.axes([0.45, 0.025, 0.1, 0.04])
    reset_button = Button(
        ax=reset_axis, 
        label='Reset', 
        hovercolor='0.975'
    )

    # the function to be called when a button is clicked
    def reset(event):
        layer_slider.reset()
        
    # register the reset function into a reset button
    reset_button.on_clicked(reset)

    # ////////////////////////////////////////////////////////////////////////////////////////////

    # show the image with slider and button
    plt.show()
    
def subplots_visualize_image(image_path: str, slice_list: list):
    
    # import nii image
    image_obj = nibabel.load(image_path)

    # extract the data as numpy array
    image_data = image_obj.get_fdata()
    
    # subplots
    figures, axes = plt.subplots(1, len(slice_list))
    for axis_index, axis in enumerate(axes):
        slice_index = slice_list[axis_index]
        axis.imshow(image_data[:, :, slice_index], cmap='gray')
        axis.set_title(f'Slice {slice_index}')
    plt.show()
    
if __name__ == '__main__':
    
    # import nii image
    image_paths = glob(os.path.join('test_prediction_images', '*.nii.gz'))
    
    
    for image_path in image_paths:
        
        # visualize image with slider GUI
        # visualize_image(image_path=image_path)
        
        # visualize image with subplots
        # slice_list = [45, 65, 85, 105, 125, 145]
        slice_list = [50, 80, 110, 140]
        subplots_visualize_image(image_path=image_path, slice_list=slice_list)