# -*- encoding: utf-8 -*-
# modeltester.py
'''
'''

from testutils import get_argv

def main(argv):
    '''
    '''
    # removing TensorFlow watning [AVX support]
    # and some Keras wrapper deprecation warning
    import os, logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

    import numpy as np
    from tensorflow.python import keras
    from imutils import paths
    from visutils.datasets import SimpleDatasetLoader
    from sklearn.metrics import classification_report
    from visutils.preprocessing import ImageToArrayPreprocessor
    from visutils.preprocessing import SimplePreprocessor
    from visutils.datasets import SimpleDatasetLoader

    imagepaths = list(
    paths.list_images(argv.dataset)
    )
    classnames = [path.split(os.path.sep)[-2] for path in imagepaths]
    classnames = [str(name) for name in np.unique(classnames)]
    print(classnames)
    
    # load images and labels
    print('[INFO] loading images...')
    sp = SimplePreprocessor(argv.image_size, argv.image_size)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(
        preprocessors=[sp, iap], grayscale = argv.gray
    )
    (data, labels) = sdl.load(
        imagepaths, verbose = 500
    )

    # load model
    modelpath = argv.model
    model = keras.models.load_model(modelpath)

    # make predictions
    predictions = model.predict(data).argmax(axis = 1)
    predictions = [classnames[prediction] for prediction in predictions]

    # make, print and write report to disk
    print('[INFO] computing inference report...')
    report = classification_report(
        labels, predictions
    )
    print(report)
    model_name = os.path.basename(argv.model)[:-3]
    with open(f'tests/output/{model_name}.txt', 'w') as f:
        f.write(report)
    


if __name__ == '__main__':
    argv = get_argv()
    main(argv)