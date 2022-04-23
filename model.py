import numpy
import torch
import torch.nn as nn

def DoubleConv3d(in_channels, out_channels):
    return nn.Sequential(        
        # 1st convolutional layer
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        # 2nd convolutional layer
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )

def crop_and_concat(target_tensor, reference_tensor):
    """
    Cropping a target tensor to have a size equals to a reference tensor,
    then concatenate the cropped target tensor to a reference tensor.

    Parameters
    ----------
    target_tensor : torch.Tensor
        A tensor desired to crop
    reference_tensor : torch.Tensor
        A tensor using as a reference of shape

    Returns
    -------
    concat_tensor : torch.Tensor
        A concatenated tensor 
    """
    
    # cropping    
    target_tensor_shape = numpy.array(target_tensor.shape[2:], dtype=int)
    reference_tensor_shape = numpy.array(reference_tensor.shape[2:], dtype=int)
    shape_half_diff = (target_tensor_shape - reference_tensor_shape) / 2
    cropped_lower_bound = numpy.floor(shape_half_diff).astype(int)
    cropped_upper_bound = numpy.floor(target_tensor_shape - shape_half_diff).astype(int)
    cropped_target_tensor = target_tensor[:, :, cropped_lower_bound[0]:cropped_upper_bound[0], cropped_lower_bound[1]:cropped_upper_bound[1], cropped_lower_bound[2]:cropped_upper_bound[2]]
    
    # concatenation
    concat_tensor = torch.cat((cropped_target_tensor, reference_tensor), dim=1)
    
    return concat_tensor

def features_list(min_feature: int = 48, features_num: int = 3):
    """
    Generate a list of features by doubling a minimum feature to obtain desired number of features

    Parameters
    ----------
    min_feature : int, optional
        Minimum feature, by default 48
    features_num : int, optional
        Number of features, by default 3

    Returns
    -------
    features : list
        List of features
    """
    
    features = [min_feature]
    for _ in range(0, features_num - 1):
        features.append(features[-1] * 2)
    print(f'Features    : {features}')
    return features
                
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[48, 96, 192, 384]):
        super(UNet, self).__init__()
        
        # layers in encoder (contracting / downward) path
        self.encoder = nn.ModuleList()
        for feature in features[:-1]:
            # double of convolutional and ReLU layers
            self.encoder.append(DoubleConv3d(in_channels=in_channels, out_channels=feature))
            # downsampling layers using max pooling
            self.encoder.append(nn.MaxPool3d(kernel_size=3, padding=1, stride=2))
            # set new value to in_channels
            in_channels = feature
        
        # layer in bottleneck (bottom) path
        # double of convolutional and ReLU layers
        self.bottleneck_conv = DoubleConv3d(in_channels=features[-2], out_channels=features[-1])
        
        # layers in decoder (expanding / upward) path
        self.decoder = nn.ModuleList()
        for feature in reversed(features[:-1]):
            # upsampling layers using transposed convolution
            self.decoder.append(nn.ConvTranspose3d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            # double of convolutional and ReLU layers
            self.decoder.append(DoubleConv3d(in_channels=feature*2, out_channels=feature))
        
        # layer in final path to produce output
        # convolutional layer with kernel_size of 1
        self.final_conv = nn.Conv3d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
        
    def forward(self, input_image):
        
        # list of layers index to replicate, crop, and concatenate
        concat_list = [];
        
        # encoder (contracting / downward) path
        x = input_image
        for layer_index, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            if layer_index % 2 == 0:
                concat_list.append(x)
        
        # bottleneck (bottom) path
        x = self.bottleneck_conv(x)
        
        # decoder (expanding / upward) path
        concat_list.reverse() # reversed the list
        for layer_index, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if layer_index % 2 == 0:
                encoded_x = concat_list[layer_index // 2]
                x = crop_and_concat(target_tensor=encoded_x, reference_tensor=x)
          
        # final convolution path
        output_image = self.final_conv(x)
        
        # tensor debugging
        # print(f'Input Shape : {input_image.shape}')
        # print(f'Output Shape: {output_image.shape}')
        
        return output_image

if __name__ == '__main__':
    
    # # parameters
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch_size = 4
    # in_channels = 1
    # out_channels = 1
    # subvoxel_dim = 64
    # features = features_list(min_feature=48, features_num=4)
    
    # # input, model, output
    # input_image = torch.rand((batch_size, in_channels, subvoxel_dim, subvoxel_dim, subvoxel_dim))
    # model = UNet(in_channels=in_channels, out_channels=out_channels, features=features)
    # output_image = model(input_image).to(device)
    
    # # tensor shape debugging
    # print(f'Input Shape : {input_image.shape}')
    # print(f'Output Shape: {output_image.shape}')
    
    pass