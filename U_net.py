import tensorflow as tf
from layers import *

def U_net(inputs, reuse = False, name = "U_net") : 

    with tf.variable_scope(name) as scope : 
        if reuse : 
            scope.reuse_variable()
        else : 
            assert scope.reuse is False
        # downsample
        conv1 = conv_2_layers(inputs, filters = 64, kernel_size=(3,3), strides=(1,1), padding="valid", n_layers=2)
        max_pool1 = max_pool(conv1, pool_size=2,strides=2)
        
        conv2 = conv_2_layers(max_pool1, filters = 128, kernel_size=(3,3), strides=(1,1), padding="valid", n_layers=2)
        max_pool2 = max_pool(conv2, pool_size=2,strides=2)

        conv3 = conv_2_layers(max_pool2, filters = 256, kernel_size=(3,3), strides=(1,1), padding="valid", n_layers=2)
        max_pool3 = max_pool(conv3, pool_size=2,strides=2)

        conv4 = conv_2_layers(max_pool3, filters = 512, kernel_size=(3,3), strides=(1,1), padding="valid", n_layers=2)
        max_pool4 = max_pool(conv4, pool_size=2,strides=2)

        conv5 = conv_2_layers(max_pool4, filters = 1024, kernel_size=(3,3), strides=(1,1), padding="valid", n_layers=2)
        
        # upsample
        upconv6 = deconvolution_layer(conv5, filters  = 512, kernel_size=(2,2), strides=(2,2), padding="valid")
        concat = crop_and_concat(conv4,upconv6)
        conv6 = conv_2_layers(concat, filters = 512, kernel_size=(3,3), strides=(1,1), padding="valid")
        
        upconv7 = deconvolution_layer(conv6, filters  = 256, kernel_size=(2,2), strides=(2,2), padding="valid")
        concat = crop_and_concat(conv3,upconv7)
        conv7 = conv_2_layers(concat, filters = 256, kernel_size=(3,3), strides=(1,1), padding="valid")
        
        upconv8 = deconvolution_layer(conv7, filters  = 128, kernel_size=(2,2), strides=(2,2), padding="valid")
        concat = crop_and_concat(conv2,upconv8)
        conv8 = conv_2_layers(concat, filters = 512, kernel_size=(3,3), strides=(1,1), padding="valid")

        upconv9 = deconvolution_layer(conv8, filters  = 64, kernel_size=(2,2), strides=(2,2), padding="valid")
        concat = crop_and_concat(conv1,upconv9)
        conv9 = conv_2_layers(concat, filters = 64, kernel_size=(3,3), strides=(1,1), padding="valid")
        outputs = convolution_layer(conv9, filters = 2, kernel_size=(1,1), strides=(1,1), padding="valid")
        
        return outputs