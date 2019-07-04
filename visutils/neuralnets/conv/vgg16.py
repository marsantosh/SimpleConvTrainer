# -*- coding: utf-8 -*-
# vgg16.py
'''
VGG (Visual Geometry Group)
Implementation of the VGG16 architecture.

/*
* To do:
* - put reference to paper
* - the width, height and depth are fixed: 224, 224, 3
*/
'''

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D



class VGG16:
    '''
    '''

    @staticmethod
    def build():
        '''
        VGG16 build method does not accepts params, the width, height
        and depth are fixed to 224, 224 and 3.
        Also the number of classes is fixed to 1000

        returns
        -------
            model: the VGG16 model compatible wiith given inputs
                as a keras sequential model
        '''
        height = 224
        width = 224
        depth = 3
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
        
        # first: CONV -*-> RELU --> layer
        model.add(Conv2D(64, (3, 3), input_shape = inputShape, padding = 'same'))
        model.add(Activation('relu'))

        # second: CONV -*-> RELU --> POOL -%-> layer
        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # third: CONV -*-> RELU --> layer
        model.add(Conv2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # fourth: CONV -*-> RELU --> POOL -%-> layer
        model.add(Conv2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # fifth: CONV -*-> RELU --> layer
        model.add(Conv2D(256, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # sixth: CONV -*-> RELU --> layer
        model.add(Conv2D(256, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # seventh: CONV -*-> RELU --> POOL -%-> layer
        model.add(Conv2D(256, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # eigth: CONV -*-> RELU --> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # nineth: CONV -*-> RELU --> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # tenth: CONV -*-> RELU --> POOL -%-> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # eleventh: CONV -*-> RELU --> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # twelveth: CONV -*-> RELU --> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # thirdteenth: CONV -*-> RELU --> POOL -%-> layer
        model.add(Conv2D(512, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # 14th: FLATTEN -||->
        model.add(Flatten())

        # 15th: FC -·-> RELU -->
        model.add(Dense(4096))
        model.add(Activation('relu'))

        # 16th: FC -·-> RELU -->
        model.add(Dense(4096))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(1000))
        model.add(Activation('softmax'))

        return model