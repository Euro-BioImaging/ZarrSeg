# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 01:00:56 2021

@author: bugra
"""

import numpy as np

###################### Two functions for creating sparsely sampled binary volumes ############################
##############################################################################################################
def create_regular_sparse(img, sparsity = (1, 2, 2)):
    """ Creates a 3D binary array of 1s and 0s, where 1s are placed with regular intervals.
        
        Parameters:
        -----------
        img: array
            A 3D numpy array or an iterable of size 3 representing array shape. If an array, the sparse array's shape
            will be img.shape. If an iterable, it represents the shape of the sparse array.
        sparsity: int (scalar or iterable of 3)
            Specifies the spacing in 3D, with which 
            the array will be sampled with 1s

        Returns: 
        --------
        out: array of 1s and 0s
            A 3D numpy array with regular 1s inserted. """
    if hasattr(img, 'nonzero'):
        if not img.ndim == 3:
            raise ValueError('This function works only with 3D images.')
        imax, jmax, kmax = img.shape
    elif hasattr(img, 'index'):
        if not len(np.array(img).ravel()) == 3:
            raise ValueError('This function works only with 3D images.')
        imax, jmax, kmax = img
    out = np.zeros((imax, jmax, kmax))
    if not ((hasattr(sparsity, 'index')) or (hasattr(sparsity, 'nonzero'))):
        if hasattr(sparsity, 'imag'):
            sparsity = [sparsity] * 3
    elif hasattr(sparsity, 'index'):
        if len(sparsity) != 3:
            raise ValueError('This function works only with 3D images.')
    elif hasattr(sparsity, 'nonzero'):
        if len(sparsity.ravel()) != 3:
            raise ValueError('This function works only with 3D images.')
    sp0, sp1, sp2 = sparsity
    out[0:imax:sp0, 0:jmax:sp1, 0:kmax:sp2] = 1
    return out



def create_random_sparse(img, sparsity = 2):
    """ Creates a 3D binary array of 1s and 0s, where 1s are placed at random.
    
        Parameters:
        -----------
        img: array
            A 3D numpy array or an iterable of size 3 representing array shape. If an array, the sparse array's shape
            will be img.shape. If an iterable, it represents the shape of the sparse array.
        sparsity: int (scalar) 
            Controls the extent of the sparsity. Higher values mean sparser representation.

        Returns: 
        --------
        out: array of 1s and 0s
            A 3D numpy array with random 1s inserted. """
    if hasattr(img, 'nonzero'):
        if not img.ndim == 3:
            raise ValueError('This function works only with 3D images.')
        imax, jmax, kmax = img.shape
    elif hasattr(img, 'index'):
        if not len(np.array(img).ravel()) == 3:
            raise ValueError('This function works only with 3D images.')
        imax, jmax, kmax = img
    shape = (imax, jmax, kmax)
    length = np.prod(shape)
    extent = length // sparsity
    randinds = np.arange(0, length, 1)
    np.random.shuffle(randinds)
    inds = randinds[:extent]
    out = np.zeros_like(randinds)
    out[inds] = 1
    return out.reshape(imax, jmax, kmax)


##############################################################################################################
##############################################################################################################