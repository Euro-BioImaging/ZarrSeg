# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:44:32 2021

@author: bugra
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


### local imports 
from .miscellaneous import *

######################################### Some filling functions #############################################
##############################################################################################################

def slicewise_fill(binary):
    """ Fills 2D holes in each 2D slice of a 3D binary object.
        
        Parameters:
        -----------        
        binary: array of bool (0s and 1s)
            A 3D numpy array with one or more binary objects.

        Returns:
        --------
        out: array 
            3D numpy array after filling. """
    cp_array(binary)
    obj = binary.copy()
    for i, item in enumerate(obj):
        obj[i] = ndi.binary_fill_holes(item)
    return resetformat(obj)


def slicewise_remove_holes(binary, hole_size):
    """ Removes 2D small holes based on a size criterion in each 2D slice of a 3D binary object.
    
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s) 
            A 3D numpy array with one or more binary objects.
        hole_size: int 
            Maximum size for the holes to be removed.         
            
        Returns:
        --------
        out: array 
            3D numpy array after hole-removal. """
    cp_array(binary)
    obj = binary > 0
    for i, item in enumerate(obj):
        obj[i] = morphology.remove_small_holes(item, hole_size)
    return resetformat(obj)


def deepfill(binary, stringency = 'low', iterations=4):
    """ An iterative filling function, which rotates volumes 90 degree and applies slicewise filling
        in each different orientation. Then the images filled in each orientation are merged based on
        a stringency criterion.

        Parameters:
        -----------
        binary: array of bool (or 1s and 0s) 
            A 3D numpy array with one or more binary objects.
        stringency: str 
            If 'low' the filling is robust but may smooth out the details in object. If 'high', the filling is 
            cautious but may leave many holes unfilled.
        iterations: int 
            Represents the number of times the filling is recursively applied.

        Returns:
        --------
        out: array of bool
            3D numpy array with same shape as binary where the binary object(s) is/are filled. """
    stack = binary.copy()
    for i in range(iterations):
        stackdep = stack.copy()
        stackrow = np.rot90(stack, axes = (0, 1))
        stackcol = np.rot90(stack, axes = (0, 2))
        filleddep = slicewise_fill(stackdep)
        filledrow = slicewise_fill(stackrow)
        filledcol = slicewise_fill(stackcol)
        rowtodep = np.rot90(filledrow, axes=(0, 1), k = 3)
        coltodep = np.rot90(filledcol, axes=(0, 2), k = 3)
        if stringency == 'high':
            stack = (filleddep & rowtodep & coltodep)
        elif stringency == 'low':
            stack = (filleddep | rowtodep | coltodep)
    out = stack.copy()
    return out


def deep_hole_remove(binary, hole_size, stringency = 'low', iterations=4):
    """ An iterative hole-removal function, which rotates volumes 90 degree and applies slicewise hole-removal
        in each different orientation. Then the images repaired this way in different orientations are merged based on
        a stringency criterion.
        
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s) 
            A 3D numpy array with one or more binary objects.
        hole_size: int 
            Maximum size for the holes to be removed. 
        stringency: str 
            If 'low' the hole-removal is robust but may smooth out the details in object. If 'high', the hole-removal is 
            cautious but may leave many holes unfilled.
        iterations: int 
            Representing the number of times the hole-removal is recursively applied.

        Returns:
        --------
            out: array of bool 
                3D numpy array with same shape as 'binary', where the small holes are removed. """
    stack = binary.copy()
    for i in range(iterations):
        stackdep = stack.copy()
        stackrow = np.rot90(stack, axes = (0, 1))
        stackcol = np.rot90(stack, axes = (0, 2))
        filleddep = slicewise_remove_holes(stackdep, hole_size)
        filledrow = slicewise_remove_holes(stackrow, hole_size)
        filledcol = slicewise_remove_holes(stackcol, hole_size)
        rowtodep = np.rot90(filledrow, axes = (0, 1), k = 3)
        coltodep = np.rot90(filledcol, axes = (0, 2), k = 3)
        if stringency == 'high':
            stack = (filleddep & rowtodep & coltodep)
        elif stringency == 'low':
            stack = (filleddep | rowtodep | coltodep)
    out = stack.copy()
    return out


def extreme_fill(binary, outer_iter = 3, inner_iter = 1, stringency = 'low'):
    """ An iterative filling-closing function, which combines two binary operations: hole-filling and morphological 
        closing in order to restore weak contours of an object. This is done via a dilation-deepfill-erosion pipeline.
        
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s)
            A 3D numpy array with one or more binary objects
        outer_iter: int 
            The number of repetitions for dilation-deepfill-erosion cycle
        inner_iter: 
            The 'iterations' parameter of the 'deepfill' function
        stringency: 
            The 'stringency' parameter of the 'deepfill' function, can be 'low' or 'high'

        Returns:
        --------
        filled: array of bool
            3D numpy array with same shape as 'binary', where the small holes are removed. """
    filled = binary.copy()
    selem = morphology.ball(1)
    for i in range(outer_iter):
        filled = morphology.binary_dilation(filled, selem)
        filled = deepfill(filled, stringency = stringency, iterations = inner_iter)
        filled = morphology.binary_erosion(filled, selem)
    return filled 


def extreme_hole_remove(binary, hole_size = 10, outer_iter = 3, inner_iter = 1, stringency = 'low'):
    """ An iterative filling-closing function, which combines two binary operations: hole-filling and morphological 
        closing in order to restore weak contours of an object. This is done via a dilation-deep_hole_remove-erosion pipeline.
        
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s)
            A 3D numpy array with one or more binary objects
        outer_iter: int
            The number of repetitions for dilation-deep_hole_remove-erosion cycle
        inner_iter: int 
            The 'iterations' parameter of the 'deep_hole_remove' function
        stringency: str 
            The 'stringency' parameter of the 'deep_hole_remove' function, can be 'low' or 'high'
        hole_size: int 
            The 'hole_size' parameter of the 'deep_hole_remove' function. Specifies maximum hole size to be removed. 

        Returns:
        --------
        filled: array of bool
            3D numpy array with same shape as 'binary', where the small holes are removed. """
    filled = binary.copy()
    selem = morphology.ball(1)
    for i in range(outer_iter):
        filled = morphology.binary_dilation(filled, selem)
        filled = deep_hole_remove(filled, hole_size, stringency = stringency, iterations = inner_iter)
        filled = morphology.binary_erosion(filled, selem)
    return filled 

##############################################################################################################
##############################################################################################################



