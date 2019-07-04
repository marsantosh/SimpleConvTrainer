# encoding: utf-8 -*-
# resnet50.py

from keras import backend as K
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform



class ResNet50:
    '''
    '''
    @staticmethod
    def _identity_block(X:'keras.layer', f:int, filters:list, stage:int, block:str):
        '''
        The identity block structure for building the ResNet50 Architecture:
        The input activation (a[i]) has the same dimension as the output activation(a[i + 2]).
        Composed of "main path" + "shortcut path":
            main path:
                + INPUT >> CONV -*-> BN -$-> RELU --> CONV -*-> BN -$-> RELU --> CONV -*-> BN -$->
            shortcut path:
                + INPUT >>
        arguments
        ---------
            X:
            f:
            filters:
            stage:
            block:
            chanDim:
        returns
        -------
            X: and identity block structure 
        '''
        # definning channel dimension axis
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension itself
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # retrieve filters
        F1, F2, F3 = filters

        # save the input value
        # we'll need this later to add back to the main path
        X_shortcut = X
        
        # first component of main path
        X = Conv2D(
            filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # second component of main path
        X = Conv2D(
            filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
            name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # third component of main path
        X = Conv2D(
            filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2c')(X)

        # final step: add shortcut value to main path and pass it 
        # throught a RELU activaiton
        X = layers.Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    
    @staticmethod
    def _convolutional_block(X:'keras.layer', f:int, filters:list, stage:int, block:str, s:int = 2):
        '''
        The convolutional block structure for building the ResNet50 Architecture:
        The input activation (a[i]) dimension does not match the output activation(a[i + 2]).
        So, the shortcut path needs a CONV -*-> layer to match this output.
        Composed of "main path" + "shortcut path":
            main path:
                + INPUT >> CONV -*-> BN -$-> RELU --> CONV -*-> BN -$-> RELU --> CONV -*-> BN -$->
            shortcut path:
                + INPUT >> CONV -*-> BN -$->
        arguments
        ---------
            X:
            f:
            filters:
            stage:
            block:
            chanDim:
        returns
        -------
            X: and identity block structure
        '''
        # definning channel dimension axis
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension itself
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # retrieve filters
        F1, F2, F3 = filters

        # save the input value
        X_shortcut = X

        # ***** main path *****
        # first component of main path
        X = Conv2D(
            filters = F1, kernel_size = (1, 1), strides = (s, s),
            name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # second component of main path
        X = Conv2D(
            filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
            name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # third component of main path
        X = Conv2D(
            filters = F3, kernel_size = (1, 1), strides = (1, 1),
            name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0)
        )(X)
        X = BatchNormalization(axis = chanDim, name = bn_name_base + '2c')(X)

        # ***** shortcut path *****
        X_shortcut = Conv2D(
            filters = F3, kernel_size = (1, 1), strides = (s, s),
            name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed = 0)
        )(X_shortcut)
        X_shortcut = BatchNormalization(axis = chanDim, name = bn_name_base + '1')(X_shortcut)

        # ***** final step *****
        # add shortcut value to main path, and pass it through a RELU activation
        X = layers.Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    
    @staticmethod
    def build(width:int, height:int, depth:int, classes:int):
        '''
        Builds the ResNet50 architecture with given width, height,
        and depth as dimensions of the input tensor and the corresponding
        number of classes of the data as the output array (tensor).

        arguments
        ---------
            width: width of input images.
            height: height of input images.
            depth: depth of input images.
            classes: number of classes of the corresponding data.
        returns
        -------
            model: 
        '''
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension itself
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        
        # define the input as a tensor
        inputs = Input(shape = inputShape)

        # zero-padding
        X = ZeroPadding2D((3, 3))(inputs)

        # stage 1
        X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), name = 'conv1')(X)
        X = BatchNormalization(axis = chanDim, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides = (2, 2))(X)

        # stage 2
        X = ResNet50._convolutional_block(
            X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [64, 64, 256], stage = 2, block = 'b'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [64, 64, 256], stage = 2, block = 'c'
        )

        # stage 3
        X = ResNet50._convolutional_block(
            X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [128, 128, 512], stage = 3, block = 'b'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [128, 128, 512], stage = 3, block = 'c'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [128, 128, 512], stage = 3, block = 'd'
        )

        # stage 4
        X = ResNet50._convolutional_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'f'
        )

        # stage 5
        X = ResNet50._convolutional_block(
            X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s = 2
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b'
        )
        X = ResNet50._identity_block(
            X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c'
        )

        # avgpool
        X = AveragePooling2D((2, 2), name = 'avg_pool')(X)

        # output layer and softmax
        X = Flatten()(X)
        X = Dense(classes, name = 'fc' + str(classes))(X)
        X = Activation('softmax')(X)

        # create model
        model = Model(inputs = inputs, outputs = X, name = 'ResNet50')

        return model