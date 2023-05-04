# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:52:33 2021

@author: bugra
"""

### local imports 
from .miscellaneous import *
from .translations import *
from skimage.measure import regionprops



############################ Bounding boxes and point-to-array transforms ####################################
##############################################################################################################

def get_bounding_box(binary, pad_width = 10, set_bounds = True):
    """ Finds a 3D object's minimum bounding box.
    
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s) 
            3D numpy array with binary object(s) to compute bounding box on.
        pad_width: int (scalar or iterable of 3) 
            The identified bounding box is padded with this parameter in each dimension.
        set_bounds: bool 
            If True, the bounding box is applied to the binary array to crop the 
            volume to the bounding box. Else returns the bounding box as a tuple.
            
        Returns: 
        --------
        cropped: array of 1s and 0s 
            If set_bounds is True, the object is cropped to the bounding box and then returned.
        bounds: tuple 
            If set_bounds is False, a tuple of size 6, which represents indices of the bounding box, is returned. """
    assert binary.ndim == 3
    if not hasattr(pad_width, 'imag'):
        pad_width = np.array(pad_width).ravel().reshape(1, 3)
    shape = np.array(binary.shape)
    pts = np.argwhere(binary)
    ptsmax = pts.max(axis = 0)
    ptsmin = pts.min(axis = 0)
    img_down = np.array([0, 0, 0])
    img_up = shape.copy()
    ptsmin_drifted = ptsmin - pad_width - 1
    ptsmax_drifted = ptsmax + pad_width
    ptsmin_drifted = np.where(ptsmin_drifted - img_down < 0, 0, ptsmin_drifted)
    ptsmax_drifted = np.where(img_up - ptsmax_drifted < 0, img_up, ptsmax_drifted)
    i0,j0,k0 = ptsmin_drifted
    i1,j1,k1 = ptsmax_drifted
    if set_bounds:
        cropped = binary[i0:i1, j0:j1, k0:k1]
        return cropped
    else:
        bounds = (i0, j0, k0, i1, j1, k1)
        return bounds


def get_bbox(binary, pad_width = None, mode = 'constant', constant_values = (0, 0)):
    """ Finds a 3D object's minimum bounding box using scikit-image's regionprops. 
        
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s)
            a 3D numpy array with binary object(s) in it.
        pad_width: scalar or iterable of size 3 
            The extent, to which the newly created bounding box is padd
        mode: str 
            mode parameter of the np.pad function
        constant_values: tuple of size 2
            constant_values parameter of the np.pad function

        Returns:
        --------
        out: array of 1s and 0s
            3D numpy array cropped to its minimum bounding box, plus padding in each dimension. """
    cp_array(binary)
    obj = resetformat(binary)
    regprops = regionprops(obj)[0]
    obj = regprops.image
    if pad_width is not None:
        obj = np.pad(obj, pad_width, mode = mode, constant_values = constant_values)
    return resetformat(obj)


def volume_from(points, pad_width = 10, scalars = None):
    """ This function calculates a bounding box to enclose the object, which is given as a volume or a point list. 
        
        Parameters:
        -----------
        points: array or list 
            Either an iterable of points or a 3D binary numpy array. If a pointset, it holds the coordinates of 
            an object in 3D space.
        pad_width: scalar or iterable of size 3 
            The extent, to which the newly created bounding box is padded.
        scalars: array or list 
            1D array or list with values corresponding to each voxel coordinate. Must have the same length as 'points'.
        
        Returns:
        --------
        out: array
            3D bounding box with the binary object in the center. If 'scalars' is not None (default None), then 
            the object's voxels are assigned the scalar values. 
        """
    if not hasattr(points, 'nonzero'):
        points = ensure2dpointlist(points)
    elif points.ndim == 3:
        points = np.argwhere(points)
    elif points.ndim < 3:
        points = ensure2dpointlist(points)
    pts = points - points.mean(axis=0)
    ptsmin = pts.min(axis = 0)
    if not np.isscalar(pad_width):
        assert len(pad_width) == 3
        pad_width = ensure2dpointlist(pad_width)
    pts = pts + np.abs(ptsmin) + pad_width
    borders = np.around(pts.max(axis = 0) + pad_width + 1).astype(int)
    borders = borders.ravel()
    out = np.zeros(borders.tolist())
    pts = np.around(pts).astype(int)
    if scalars is None:
        out[tuple(pts.T)] = 1
    else:
        out[tuple(pts.T)] = scalars
    return out     


def coords_to_array(coords, array, scalars = None): ### this function fits the given indices inside an empty array by deleting indices that don't fit
    """ A simple wrap around the "translations" module's "fit_coords_to_array" function. The aim is to
        simply make this frequently-used function accessible from both modules.
        
        Parameters:
        -----------
        coords: array or list (list of voxel coordinates) 
            A 2D numpy array or list, the columns of which must be of length 3. 
        array: array or iterable of size 3  
            Either a 3D numpy array or an iterable of size 3. If a 3D array, the new array to store the 'coords' in 
            will be created by assuming the shape of this array. If an iterable, the values of this iterable will be 
            used as the shape of the new array.
                
        Returns:
        --------
        out: array of 1s and 0s
            The newly created 3D array with value 1 for points included in the 'coords' and 0 for points not included.'
        """
    out = fit_coords_to_array(coords, array, scalars)  
    return out


c2r = lambda coords, array: coords_to_array(coords, array) ### wrap around tl.fit_coords_to_array for abbreviation

##############################################################################################################
##############################################################################################################

