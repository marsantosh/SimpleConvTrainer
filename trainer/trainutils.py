# -*- encoding: utf-8 -*-
# trainutils.py

import argparse
import datetime



def get_argv():
    '''
    Get the program argument values.

    returns
    -------
        argv: an object (Namespace) with the parsed arguments
            as attributes.
    '''
    ap = argparse.ArgumentParser(
        prog = 'python(3.6) trainer/main.py', description = '''
        Convolutional Neural Network trainer for a specific architecture using
        TensorFlow as the backend engine and OpenCV for vision utilities.
            -msantosh@axtellabs
        '''
    )
    ap.add_argument(
        '-a', '--architecture', required = True, type = str,
        help = 'the architecture of the cnn to be trained. Options: `lenet`, \
            `minivgg`, `karpathynet`.'
    )
    ap.add_argument(
        '-d', '--dataset', required = True, type = str,
        help = 'path to input dataset'
    )
    ap.add_argument(
        '-m', '--model', required = True, type = str,
        help = 'name [syntax] of output model file'
    )
    ap.add_argument(
        '-g', '--grayscale', default = False, type = bool,
        help = 'load images in grayscale or not (default is False)'
    )
    ap.add_argument(
        '-i', '--image-size', default = 64, type = int,
        help = 'the size of the images in pixels for the width and height'
    )
    ap.add_argument(
        '-o', '--optimizer', default = 'SGD', type = str,
        help = 'the optimizer to use for the gradient descent variant step \
        for optimization. Supported optimizers: `adam` and `sgd`'
    )
    ap.add_argument(
        '-decay', '--lr-decay', default = False, type = bool,
        help = 'lapply time based learning rate < decay_rate = learning_rate / epochs >'
    )
    ap.add_argument(
        '-etha', '--etha', default = 0.001, type = float,
        help = 'alpha (or etha) the learning rate for optmizer during the gradient descent step'
    )
    ap.add_argument(
        '-e', '--epochs', default = 30, type = int,
        help = 'number of epochs to train the network'
    )
    ap.add_argument(
        '-b', '--batch-size', default = 32, type = int,
        help = 'the size of the batch to train with stochastic gradient descent'
    )
    ap.add_argument(
        '-do', '--dropout', default = False, type = bool,
        help = 'whether or not apply dropout to the `KarpathyNet` architecture after the MaxPooling layers.'
    )

    argv = ap.parse_args()

    return argv



def build_architecture(argv, width:int, height:int, depth:int, classes:int):
    '''
    Build the chosen architecture model.

    arguments
    ---------
        argv: the argument values the program was called with
        width: width of input images
        height: height of the input images
        depth: depth (no. of channels) of the input images
        classes: number of classes of the corresponding data
    returns
    -------
        model: the correspondant keras implementation model
            compatible with the given inputs.
    '''
    if argv.architecture == 'lenet':
        from visutils.neuralnets.conv import LeNet
        model = LeNet.build(
        width = width,
        height = height,
        depth = depth,
        classes = classes
        )
    
    elif argv.architecture == 'karpathynet':
        from visutils.neuralnets.conv import KarpathyNet
        model = KarpathyNet.build(
        width = width,
        height = height,
        depth = depth,
        classes = classes,
        dropout = argv.dropout
        )
    
    elif argv.architecture == 'minivgg':
        from visutils.neuralnets.conv import MiniVGGNet
        model = MiniVGGNet.build(
        width = width,
        height = height,
        depth = depth,
        classes = classes
        )
    
    return model



def load_images_and_labels(argv, imagepaths):
    '''
    '''
    from visutils.preprocessing import ImageToArrayPreprocessor
    from visutils.preprocessing import AspectAwarePreprocessor
    from visutils.datasets import SimpleDatasetLoader
    # initializing the image preprocessors
    imagesize = argv.image_size
    aap = AspectAwarePreprocessor(imagesize, imagesize)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk to memory
    # then normalize raw pixel intensities
    print('[INFO] loading images...')
    sdl = SimpleDatasetLoader(
        preprocessors = [aap, iap], grayscale = argv.grayscale
    )
    (data, labels) = sdl.load(
        imagepaths, verbose = 500
    )
    print('[INFO] normalizing data...')
    data = data.astype('float') / 255.0

    return data, labels



def write_training_report(argv, modelparams, report):
    '''
    '''
    imagesize = modelparams['imagesize']
    with open(f'output/reports/{argv.model}_report.txt', 'w') as f:
        f.write(f"REPORT FOR MODEL: {argv.model}\n\n")
        f.write(f"DATETIME: {datetime.datetime.now()}\n\n")
        
        f.write("             ========   PROGRAM ARGUMENTS   ========\n\n")
        for arg, val in argv._get_kwargs():
            f.write(f'{arg + ":":12}    {val}\n')
        f.write("\n\n")
        f.write("             ======== MODEL HYPERPARAMETERS ========\n\n")
        f.write(f"image size:\t{str(imagesize)} * {str(imagesize)}\n")
        f.write(f"epochs:\t{argv.epochs:10}\n")
        f.write(f"optimizer:\t{argv.optimizer:10}\n\n")
        
        optconfig = modelparams['optimizer'].get_config()
        for ft in optconfig:
            f.write(f"\t\t\t* {ft:10}: \t{optconfig[ft]:10}\n")
        f.write(f"\nloss function:\t{modelparams['loss_function']:10}\n")
        f.write(f"metrics:\t{str(modelparams['metrics']):10}\n\n\n")
        f.write("             ========    ELAPSED TRAINING TIME ========\n\n")
        f.write(f"elapsed training time:\t{modelparams['elapsed_time'] / 60:10.3f} minutes\n\n\n")
        f.write("             ========     MODEL SUMMARY  ========\n\n")
        modelparams['model_structure'].summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        f.write("             ======== CLASSIFICATION REPORT ========\n\n")
        f.write(report)
        f.write("\n\n")
