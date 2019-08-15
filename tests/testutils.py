# -*- encoding: utf-8 -*-
# testutils.py
'''
'''

import argparse


def get_argv():
    '''
    Get the program argument values.

    returns
    -------
        argv: an object (Namespace) with the parsed arguments
            as attributes.
    '''
    
    
    def string2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    description = '''
    A simple python program to test a keras model within a test set.
    '''
    ap = argparse.ArgumentParser(
        prog = 'python(3.6) tests/modeltester.py', formatter_class = argparse.RawDescriptionHelpFormatter,
        description = description
    )
    ap.add_argument(
        '-d', '--dataset', required = True, type = str,
        help = 'path to input (test) dataset.'
    )
    ap.add_argument(
    '-m', '--model', required = True, type = str,
    help = 'path to the model output (h5 keras file)'
    )
    ap.add_argument(
    '-i', '--image-size', required = True, type = int,
    help = 'the size of the width of image (and height) of the input tensor to test'
    )
    ap.add_argument(
    '-g', '--gray', default = False, type = string2bool,
    help = 'load the images in grayscale or not'
    )
    argv = ap.parse_args()

    return argv


