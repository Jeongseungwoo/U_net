import tensorflow as tf

def convolution_layer(inputs, filters, kernel_size = (3,3), strides = (1,1), padding = "valid") :
    conv = tf.layers.conv2d(inputs = inputs, 
                            filters = filters, 
                            kernel_size = kernel_size, 
                            strides = strides, 
                            padding = padding, 
                            activation=tf.nn.relu)
    return conv

def conv_2_layers(inputs, filters, kernel_size = (3,3), strides = (1,1), padding = "valid", n_layers = 2) :
    for i in range(n_layers) :
        conv = convolution_layer(inputs = inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        inputs = conv
    return conv

def deconvolution_layer(inputs, filters, kernel_size = (2,2) ,strides = (2,2),padding="valid") :
    deconv = tf.layers.conv2d_transpose(inputs = inputs, 
                                        filters = filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides,
                                        padding = padding, 
                                        activation = None)
    return deconv

def max_pool(inputs, pool_size = 2, strides = 2) :
    max_pool = tf.layers.max_pooling2d(inputs = inputs, pool_size = pool_size, strides=strides)
    return max_pool

def crop_and_concat(input_A, input_B) :
    
    A_shape = input_A.get_shape()
    B_shape = input_B.get_shape()
    
    offsets = [0, int((A_shape[1]-B_shape[1])//2), int((A_shape[2]-B_shape[2])//2), 0]
    size = [-1, int(B_shape[1]), int(B_shape[2]),-1]
    crop = tf.slice(input_A,offsets,size)
    
    concat = tf.concat([crop, input_B],axis=3)
    return concat