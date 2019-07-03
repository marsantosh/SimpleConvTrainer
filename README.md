# SimpleConvTrainer

A simple CNN trainer optimized for vision purpousses.

Available CNN architectures:
```
- lenet
- minivgg
- karpathynet
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
usage: python(3.6) trainer/main.py [-h] -a ARCHITECTURE -d DATASET -m MODEL
                                   [-g GRAYSCALE] [-i IMAGE_SIZE]
                                   [-o OPTIMIZER] [-decay LR_DECAY]
                                   [-etha ETHA] [-e EPOCHS] [-b BATCH_SIZE]
                                   [-do DROPOUT]

Convolutional Neural Network trainer for a specific architecture using
TensorFlow as the backend engine and OpenCV for vision utilities.
-msantosh@axtellabs

optional arguments:
  -h, --help            show this help message and exit
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        the architecture of the cnn to be trained. Options:
                        `lenet`, `minivgg`, `karpathynet`.
  -d DATASET, --dataset DATASET
                        path to input dataset
  -m MODEL, --model MODEL
                        name [syntax] of output model file
  -g GRAYSCALE, --grayscale GRAYSCALE
                        load images in grayscale or not (default is False)
  -i IMAGE_SIZE, --image-size IMAGE_SIZE
                        the size of the images in pixels for the width and
                        height
  -o OPTIMIZER, --optimizer OPTIMIZER
                        the optimizer to use for the gradient descent variant
                        step for optimization. Supported optimizers: `adam`
                        and `sgd`
  -decay LR_DECAY, --lr-decay LR_DECAY
                        lapply time based learning rate < decay_rate =
                        learning_rate / epochs >
  -etha ETHA, --etha ETHA
                        alpha (or etha) the learning rate for optmizer during
                        the gradient descent step
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the network
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the size of the batch to train with stochastic
                        gradient descent
  -do DROPOUT, --dropout DROPOUT
                        whether or not apply dropout to the `KarpathyNet`
                        architecture after the MaxPooling layers.
```

## OUTPUTS
The program will output three files
+ A {.h5} file in ```output/model/{filename}``` which represents the keras
compatible CNN model.
+ A {.txt} file in ```output/report/{filename}``` which represents the training model
report and the test validation information.
+ A {.png} file in ```output/plot/{filename}``` which represents the training plot
metrics monitor over the training phase.