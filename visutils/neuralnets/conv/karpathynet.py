# -*- coding: utf-8 -*-
# karpathynet.py
'''
Implementation of the KarpathyNet architecture in Keras
'''


from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D



class KarpathyNet:
    '''
    KarpathyNet Architecture implementation in Keras
    '''

    @staticmethod
    def build(width:int, height:int, depth:int, classes:int, dropout:bool = False):
        '''
        Build the KarpathyNet architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.
        
        arguments
        ---------
            width:  width of input images.
            height: height of input images.
            depth:  depth if input images.
            classes: number of classes of the corresponding data.
            dropout: whether to apply dropout to MaxPooling layers
                or not.
        returns
        -------
            model: the KarpathyNet model compatible with given inputs
                    as a keras sequential model.
        '''
        # initialize model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV -*-> RELU --> POOL layers
        model.add(Conv2D(16, (5, 5), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        if dropout:
            model.add(Dropout(0.25))

        # second set of CONV -*-> RELU --> POOL layers
        model.add(Conv2D(20, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        if dropout:
            model.add(Dropout(0.25))
        
        # third set of CONV -*-> RELU --> POOL layers
        model.add(Conv2D(20, (5, 5), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        if dropout:
            model.add(Dropout(0.25 * 2))

        # first (and only) set of FC -·-> DENSE layers
        model.add(Flatten())
        model.add(Dense(classes))

        # softmax classifier
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model