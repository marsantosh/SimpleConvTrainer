# -*- encoding: UTF-8 -*-
# main.py

from trainutils import get_argv
from trainutils import write_training_report
from trainutils import load_images_and_labels
from trainutils import build_architecture

def main(argv):
    '''
    '''
    # print program arguments
    print('[INFO] program arguments: ')
    for arg, val in argv._get_kwargs():
        print(f'  {arg + ":":12}    {val}')
    
    print('[INFO] setting up program...')
    print('[INFO] importing libraries...')
    
    # removing TensorFlow watning [AVX support]
    # and some Keras wrapper deprecation warning
    import os, logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

    # importing libraries
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam, SGD
    from keras.utils import to_categorical
    from keras.callbacks import ModelCheckpoint
    from imutils import paths
    from visutils.callbacks import TrainingPlot, EpochCheckpoint

    # extracting the class labels from image paths
    print('[INFO] extracting class names...')
    imagepaths = list(
        paths.list_images(argv.dataset)
    )
    classnames = [path.split(os.path.sep)[-2] for path in imagepaths]
    classnames = [str(name) for name in np.unique(classnames)]

    # load the images and their respective labels
    data, labels = load_images_and_labels(
        argv, imagepaths, normalize = argv.normalize
    )

    # partition dataset into training and testing
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size = 0.10, random_state = 42
    )

    # reshaping data
    if argv.grayscale:
        imagesize = argv.image_size
        print("[INFO] reshaping data to fit grayscale format...")
        trainX = trainX.reshape(trainX.shape[0], imagesize, imagesize, 1)
        testX = testX.reshape(testX.shape[0], imagesize, imagesize, 1)
    print("[INFO] data shape:")
    print("   trainX.shape: {}\n   testX.shape: {}".format(trainX.shape, testX.shape))
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print("   trainY: {}".format(trainY.shape))
    print("   testY: {}".format(testY.shape))
    print("   testY[0]: {}".format(testY[0]))

    # construtct iamge generator object for data augmentation
    augmentation = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    # initialize the optimizer and model
    # the neural net architecture input size will have
    # magnitudes of width and height and channel dimension,
    # according to the input image.
    print('[INFO] compiling model...')
    if argv.optimizer == 'adam':
        optimizer = Adam(lr = argv.etha)
    else:
        optimizer = SGD(lr = argv.etha)
    
    # setting callbacks
    modelpath = f'output/models/{argv.model_name}.h5'
    plotpath = f'output/plots/{argv.model_name}.png'
    callbacks = [
        ModelCheckpoint(
            filepath = modelpath,
            save_best_only = True
        ),
        TrainingPlot(
            figPath = plotpath
        )
    ]

    # building model
    # getting width, height and depth of images
    __, width, height, depth = testX.shape
    model = build_architecture(
        argv,
        width = width,
        height = height,
        depth = depth,
        classes = len(classnames)
    )

    # compiling model
    loss_function = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(
        loss = loss_function,
        optimizer = optimizer,
        metrics = metrics
    )

    print("[INFO] LeNet Input Structure:")
    print(
        f"   width: {width}, height: {height}, depth: {depth}, classes: {len(classnames)}", end = '\n'
    )

    # fitting model
    t = time.time()
    print('[INFO] training network:')
    print('[INFO] launching TensorFlow engine:')
    H = model.fit_generator(
        augmentation.flow(trainX, trainY,
        batch_size = argv.batch_size),
        validation_data = (testX, testY),
        steps_per_epoch = len(trainX) // argv.batch_size,
        epochs = argv.epochs,
        callbacks = callbacks,
        verbose = 1
        )
    elapsed_time = time.time() - t

    # evaluating the network
    print("[INFO] evaluating network...")
    predictions = model.predict(
        testX, batch_size = argv.batch_size
    )
    report = classification_report(
        y_true = testY.argmax(axis = 1),
        y_pred = predictions.argmax(axis = 1),
        target_names = classnames
    )

    # write model report
    # the report will be written in the path:
    # output/reports/{model_name}_report.txt
    print("[INFO] writing training report...")
    modelparams = {
        'imagesize': argv.image_size,
        'optimizer': optimizer,
        'model_structure': model,
        'loss_function': loss_function,
        'metrics': metrics,
        'elapsed_time': elapsed_time
    }
    write_training_report(argv, modelparams, report)


if __name__ == '__main__':
    argv = get_argv()
    main(argv)