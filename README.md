# SimpleConvTrainer

A simple CNN trainer optimized for vision purpousses.

Available CNN architectures:
```
- lenet
- minivgg
- karpathynet
- alexnet
- vgg16 (fixed size)
```

## Requirements

Required Python libraries:
+ numpy
+ TensorFlow
+ Keras
+ OpenCV
+ scikit-learn
+ scikit-image
+ imutils
+ imageio

## USAGE
First source the environment with
```
source env.sh
```
this will activate the vision environment and
initialize the `PYTHONPATH`.

To see program instructions:
```
python trainer/main.py --help
````
```
usage: python(3.6) trainer/main.py [-h] -a ARCHITECTURE -d DATASET
                                   [-n NORMALIZE] -mn MODEL_NAME
                                   [-g GRAYSCALE] [-i IMAGE_SIZE]
                                   [-o OPTIMIZER] [-etha ETHA] [-e EPOCHS]
                                   [-b BATCH_SIZE]

Convolutional Neural Network trainer for a specific architecture using
TensorFlow as the backend engine and OpenCV for vision utilities.
                                        -msantosh@axtellabs.
The supported architectures are:
    - MiniVGG
    - LeNet
    - AlexNet
    - KarpathyNet
    - VGG16 (fixed size input {224, 224, 3} and 1000 classes.)

All architectures (until now) can be specified their input shapes
and their output dimension (number of classes), except for the VGG16.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        the architecture of the cnn to be trained. Options:
                        `lenet`,`minivgg`,`karpathynet`.
  -d DATASET, --dataset DATASET
                        path to input dataset.
  -n NORMALIZE, --normalize NORMALIZE
                        normalize the input images intensities or not (default
                        is True).
  -mn MODEL_NAME, --model-name MODEL_NAME
                        name [syntax] of output model file.
  -g GRAYSCALE, --grayscale GRAYSCALE
                        load images in grayscale or not (default is False).
  -i IMAGE_SIZE, --image-size IMAGE_SIZE
                        the size of the images in pixels (width and height)
                        for the input tensor.
  -o OPTIMIZER, --optimizer OPTIMIZER
                        the optimizer to use for the gradient descent variant
                        step for optimization. Supported optimizers: `adam`
                        and `sgd`.
  -etha ETHA, --etha ETHA
                        the learning rate for optmizer during the gradient
                        descent step.
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the network.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the size of the batch to train with stochastic
                        gradient descent.
```

For running examples see: ```run_training_loop.sh``` and ```run_training_single.sh```

## OUTPUTS
The program will output three files
+ A {.h5} file in ```output/model/{filename}``` which represents the keras
compatible CNN model.
+ A {.txt} file in ```output/report/{filename}``` which represents the training model
report and the test validation information.
+ A {.png} file in ```output/plot/{filename}``` which represents the training plot
metrics monitor over the training phase.