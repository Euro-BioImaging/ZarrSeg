# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:58:23 2019

@author: ozdemir
"""

""" This module contains convenience functions for image processing. These include function for changing array data type, 
    finding bounding boxes, hole-filling, labelling, separation of labeled object etc.
    """



import numpy as np
from scipy import ndimage as ndi
from skimage.draw import ellipsoid



##################################### Some miscellaneous functions ###########################################
##############################################################################################################

def ensure2dpointlist(ptlist, ndim = 3):
    """ Ensures that a given array-like input is a 2D numpy array with
        its column size equal to 'ndim' parameter. """
    if ndim is None:
        assert not np.isscalar(ptlist[0]), 'If "ndim" is None, then "ptlist" must be at least 2D.'
        ndim = len(ptlist[0])
    serialised = np.array(ptlist).ravel()
    assert len(serialised) % ndim == 0, 'Point number is indivisible by the required dimensionality.'
    return serialised.reshape(-1, ndim)


def check_dataform(data, return_pointlist = True, ndim = None):
    if not hasattr(data, 'nonzero'):
        data = np.array(data)
    if data.ndim == 1:
        form = 0 # unstructured data
    elif data.ndim == 2:
        if data.shape[1] <= 5: 
            form = 1 # coordinate list
        if data.shape[1] > 5:
            form = 2 # object 
    elif data.ndim == 3:
        form = 2
    if return_pointlist:
        if form == 2:
            out = np.argwhere(data)
        elif form < 2:
            out = ensure2dpointlist(data, ndim = ndim)
    else:
        out = form
    return out            
            
            


def div_nonzero(im0, im1):
    """ Convenience function to divide two images voxelwise. Handles the zero-division cases. """
    if not np.isscalar(im0):
        eps = np.finfo(im0.dtype).eps
    else:
        eps = np.finfo(im1.dtype).eps
    denom = im1.copy()
    denom[denom == 0] = eps
    return im0 / denom


def cp_array(arrlike):
    """ Checkpoint: raise error if the input is not a numpy array. """
    if not hasattr(arrlike, 'nonzero'):
        raise ValueError('Only numpy arrays are supported.')
    
def cp_3d(arrlike):
    """ Checkpoint: raise error if the input is not a 3D numpy array. """
    cp_array(arrlike)
    if arrlike.ndim != 3:
        raise ValueError('Only 3D numpy arrays are supported while input is {}-dimensional'.format(arrlike.ndim))
    

def resetformat(img):
    """ Converts any numpy array into a binary array consisting of 1s and 0s. 
        If the 'img' is numerical, all voxels with value above zero will be changed to 1. 
        If the 'img' is boolean, all True voxels will be changed to 1, while the False ones to 0.
        
        Parameters:
        -----------
        img: array 
            Any numpy array, whether gray-valued or boolean.

        Returns:
        --------        
        reset: array of int
            An array consisting of 1s and 0s. """
    cp_array(img)
    reset = np.where((img > 0) | (img == True), 1, 0)
    reset = reset.astype(int)
    return reset


def eselem(a, b, c = None, d = None):
    """ Creates an ellipsoidal structuring element in 2D, 3D and, in a specialised manner, in 4D. """
    if c is None:
        selem = ellipsoid(1, a, b)[2:-2, 1:-1, 1:-1][0]
    elif d is None:
        selem = ellipsoid(a, b, c)[1:-1, 1:-1, 1:-1]
    else:
        selem = ellipsoid(b, c, d)[1:-1, 1:-1, 1:-1]
        selem = np.array(a * [selem])
    return selem


def cartesian(*arrays):
    """ A fast way to get cartesian product of given iterables. 
        
        Note: This code was copied from Stackoverflow.    
    
        Parameters:
        -----------
        arrays: iterable
            Arrays, of which the cartesian product is calculated.
            
        Returns:
        --------
        arr: array
            2D numpy array giving the cartesian product.
        """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def create_surface(binary, thickness = 1, spacing = (1, 1, 1)):
    """ Calculates the surface of a binary object using its euclidean distance transform.
    
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s)
            A 3D binary object, of which the surface is computed.
        thickness: int (scalar)
            Controls the thickness of the calculated surface.
        spacing: int (iterable of 3) 
            The spacing parameter representing the voxel size for each dimension. Used in the
            distance transform function.

        Returns:
        --------
        out: array of 1s and 0s 
            Calculated surface of the binary object. """    
    dmap = ndi.distance_transform_edt(binary, spacing)
    out = resetformat((dmap <= thickness) & (dmap > 0))
    return out


def rescale(data, low = 0, high = 1):
    """ Minmax normalisation function, which linearly rescales an image's intensities between a user-defined low and high.
        
        Parameters:
        -----------
        data: array
            array whose values will be rescaled.
        low: float
            lower bound for the rescaled values.
        high: float
            higher bound for the rescaled values.
        
        Returns:
        --------
        rescaled: array
            array with same size as data and with rescaled values.
        """
    rescaled = low + ((data - data.min()) * (high - low)) / (data.max() - data.min())
    return rescaled


def standardise(im):
    """ Normalisation that ensures an array's mean is zero and standard deviation is 1.
        
        Parameters:
        -----------
        data: array
            array whose values will be standardised.
        
        Returns
        --------
        normalised: array
            array with same size as data and with standardised values.
        """
    avg = np.mean(im); stdev = np.std(im)
    return (im - avg) / stdev


##############################################################################################################
##############################################################################################################

    






















    
    
    
    
