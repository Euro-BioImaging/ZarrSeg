# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:53:44 2019

@author: ozdemir
"""


""" This module contains several convenience functions that mostly perform translational operations on binary objects.
    Functions: 
        fit_to_coords: a convenience function that fits any given set of coordinates to a 3D numpy array. 
        geo_transform: a function that applies a given transformation matrix on a given set of coordinates.
        fit_coords_to_array: a convenience function that fits an object represented in the form of a coordinate list 
                            to a 3D array of either a given size or a size matching the dimensions of the 'array' parameter
        translate_by_center: a convenience function that displaces a 3D object to a given location so that the centroid of 
                            the object sits at the location.
        translate_to_location: a convenience function that displaces a 3D object to a given location within a given 3D space.
        translate_to_center: a convenience function that displaces a 3D object to the cenral coordinate in the same 3D space 
                            where the object resides. 
    """


import numpy as np
# from .miscellaneous import *


def fit_to_coords(coords, s=0): ## convert coordinate points to 3d image of minimal size
    """ Fits a given 2D coordinate list to a 3D image of minimal size. 
    Parameters:
        coords: a 2D numpy array or list, the columns of which must be of length 3.
        s: scalar, if nonzero, the returned array is illustrated.
    Returns:
        3D numpy array, which has value 1 for points included in the coords list and 0 for points not included. """
    if not hasattr(coords, 'nonzero'):
        coords = np.array(coords)
    origined = coords-coords.mean(axis=0)
    positived = origined-origined.min(axis=0)
    dep,row,col = positived.max(axis=0)+1
    matrix = np.zeros([int(i) for i in [dep,row,col]])
    coords_to_fit = tuple(positived.T.astype(np.int64))
    matrix[coords_to_fit] = 1
    return matrix


def geo_transform(coords, tfm_mat):
    """ Transforms an object represented the form of a coordinate list with a given transformation matrix.
        The matrices here include mainly geometric transformations such as translation, rotation, etc.
    Parameters:
        coords: a 2D numpy array or list, the columns of which must be of length 3.
        tfm_mat: the transformation matrix.
    Returns:
        3D numpy array, which is created by fitting the resulting transformed coordinates to minimal 3D volume. """
    if not hasattr(coords, 'nonzero'):
        coords = np.array(coords)
    topad = np.ones(len(coords))
    padded_coords = np.vstack([coords.T,topad])
    transformed = (tfm_mat@padded_coords).T
    binary = fit_to_coords(transformed[:,:3],s=0)
    return binary


def fit_coords_to_array(coords, array, scalars = None): ### this function fits the given indices inside an empty array by deleting indices that don't fit
    """ A convenience function that fits an object represented in the form of a coordinate list 
            to a 3D array of either a given size or a size matching the dimensions of the 'array' parameter.
            Coordinates that are present in the 'coords' parameters but that do not fit in the target array 
            are simply deleted.
    Parameters:
        coords: a 2D numpy array or list, the columns of which must be of length 3. 
        array: either a 3D numpy array or an iterable of size 3. If a 3D array, the new array to store the 'coords' in 
            will be created by assuming the dimensions of this array. If an iterable, the values of this iterable will be 
            used as the dimension lengths of the new array.
    Returns:
        the newly created 3D array which has either value 1, or given scalars, for points included in the 
            'coords' and 0 for points not included.'
        """
    if not hasattr(coords, 'nonzero'):
        coords = np.array(coords)
    if hasattr(array, 'nonzero'):
        if array.ndim == 3:
            dim0, dim1, dim2 = np.array(array.shape)
        elif array.ndim != 3:
            if len(array) == 3:
                dim0, dim1, dim2 = array
            elif len(array) > 3:
                raise ValueError('This function works only with 3D objects.')
    elif type(array) == type(()) or type(array) == type([]):
        dim0,dim1,dim2 = array
    dep,row,col = tuple(coords.T)
    depmask0 = coords[:, 0] >= 0
    rowmask0 = coords[:, 1] >= 0
    colmask0 = coords[:, 2] >= 0
    depmask1 = coords[:, 0] < dim0
    rowmask1 = coords[:, 1] < dim1
    colmask1 = coords[:, 2] < dim2
    mask = depmask0 & depmask1 & rowmask0 & rowmask1 & colmask0 & colmask1
    selected_coords = coords[mask].astype(int)
    if scalars is not None:
        scalars = scalars[mask]
    newarray = np.zeros((dim0, dim1, dim2))
    if scalars is None:
        newarray[tuple(selected_coords.T)] = 1
    else:
        newarray[tuple(selected_coords.T)] = scalars
    return newarray



def translate_by_center(obj, location):
    """ Displaces a 3D object to a given location so that the centroid of the object sits at the location. 
    Parameters:
        obj: either a 3D numpy array or a 2D pointlist representation of it (numpy array or list with a column length of 3).
        location: an iterable of size 3, the coordinates of the destination.
    Returns:
        2D pointlist representation of the displaced objects. """
    if not hasattr(location, 'nonzero'):
        location = np.array(location)
    if len(location.flatten()) < 3:
        raise ValueError('Only 3D volumes are supported in this function, so 3 coordinate points must be provided.')
    if not hasattr(obj, 'nonzero'):
        obj = np.array(obj)
    if obj.ndim == 2:
        coords = obj.copy()
    elif obj.ndim == 3:
        coords = np.argwhere(obj)
    centre = coords.mean(axis=0)
    displacement = location-centre
    displaced = coords+displacement
    displaced = displaced.astype(int)
    return displaced



def translate_to_location(obj, location, space):
    """ Displaces a 3D object to a given location within a given 3D space. 
    Parameters:
        obj: either a 3D numpy array or a 2D pointlist representation of it (numpy array or list with a column length of 3).
        location: an iterable of size 3, the coordinates of the destination.
        space: either a 3D numpy array representing the space, within which the object is translated to a given position, 
            or an iterable of 3, which provide the dimensions for creation of an empty 3D volume to translate the object to.  
    Returns:
        A 3D numpy array with the object inserted at the desired location. """
    if not hasattr(space, 'nonzero'):
        space = np.array(space)
    if space.ndim != 3:
        if len(space) == 3:
            space = np.zeros(space)
        else:
            raise ValueError('The buffer array must be a 3D numpy array.')        
    located_coords = translate_by_center(obj, location)
    return fit_coords_to_array(located_coords, space)




def translate_to_center(obj):
    """ Displaces a 3D object to the cenral coordinate in the same 3D space where the object resides. 
    Parameters:
        obj: either a 3D numpy array or a 2D pointlist representation of it (numpy array or list with a column length of 3).
    Returns:
        A 3D numpy array of the same size as the object, with the object translated to the centre. """
    translated = translate_by_center(obj, location = np.array(obj.shape)//2)
    newarr = np.zeros_like(obj)
    newarr[tuple(translated.T)] = 1
    return(newarr)



