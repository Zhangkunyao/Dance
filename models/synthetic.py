"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging
import skimage
import skimage.transform



def np_onehot(label, num_classes):
    return np.eye(num_classes)[label]


def augment_label(label, num_classes, scale=8, keep_prop=0.8):
    """
    Add noise to label for synthetic benchmark.
    """
    label = label.squeeze(1)
    label = label.cpu().detach().numpy()
    label = (label+1)/2
    label = np.round(label)
    label = np.uint8(label)
    # shape = label.shape
    # label = label.reshape(shape[-1], shape[-2])

    onehot = np_onehot(label, num_classes)
    onehot = onehot.transpose(0, 3, 1, 2)
    # lower_shape = (shape[-1] // scale, shape[-2] // scale)
    #
    # label_down = skimage.transform.resize(
    #     onehot, (lower_shape[0], lower_shape[1], num_classes),
    #     order=1, preserve_range=True, mode='constant')
    #
    # onehot = skimage.transform.resize(label_down,
    #                                   (shape[-1], shape[-2], num_classes),
    #                                   order=1, preserve_range=True,
    #                                   mode='constant')
    return onehot

