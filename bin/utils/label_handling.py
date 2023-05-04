# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 01:02:40 2021

@author: bugra
"""

from .miscellaneous import *
from skimage.measure import regionprops_table
from skimage import morphology
import pandas as pd

################################ Functions related to label image ############################################
##############################################################################################################

def expand_labels(label_image, distance = 1, sampling = (2, 1, 1)):
    """ Dilates the labels in an image in such a way that
        the labels that meet each other stop expanding, so labels with higher values
        do not overgrow on labels with lower values. 
        
        NOTE: This function was directly copied from scikit-image's function with the same name: 
            https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_expand_labels.py#L16-L106 """
    distances, nearest_label_coords = ndi.distance_transform_edt(label_image == 0, 
                                                                 sampling = sampling,
                                                                 return_indices = True)
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [dimension_indices[dilate_mask]
                                   for dimension_indices in nearest_label_coords]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def label_expansion(lbld, distance = 1, sampling = (1, 1, 1)):
    """ A 4D extension of label expansion. Each volume in a 4D data will be subjected to 'expand_labels' function. """
    diff = 0
    if lbld.ndim < 4:
        diff = 4 - lbld.ndim
    for i in range(diff):
        lbld = np.array([lbld])
    return np.array([expand_labels(item, distance, sampling) for item in lbld])


def get_dmap(imser, sampling = (2, 1, 1)):
    """ A 4D extension of the euclidean distance transform function from SciPy. 
        Each volume in the 4D image is separately subjected to the distance transform. """
    if imser.ndim == 3:
        imser = imser.reshape(1, *imser.shape)
    if len(sampling) != 3:
        raise ValueError('sampling must be a tuple of size 3 corresponding to the three image dimensions.')
    newim = []
    for item in imser:
        dmap = ndi.distance_transform_edt(item, sampling)
        newim.append(dmap)
    return np.array(newim)
        

def hierarchical_label(ser, verbosity = False):
    """ A labelling function adapted to 4D images. Each object in the 4D data is
        labelled based on the 3D connectivity. The labelling ensures that each next
        volume's minimum label equals the current volume's maximum label + 1. 
        
        This function also handles 3D images, and thus can be used as a generic labelling function. """
    assert ser.ndim > 2
    dim_changed = 0
    if ser.ndim < 4:
        ser = ser.reshape(1, *ser.shape)
        dim_changed = 1
    if len(np.unique(ser)) == 2:
        ser = np.array([morphology.label(item) for item in ser])
    maxima = [item.max() for item in ser]
    minima = [item[item > 0].min() for item in ser]
    maxima = [0] + maxima
    minima = minima + [0]
    maxcumul = np.cumsum(maxima)[:-1]
    mincumul = np.cumsum(minima)[:-1]
    cumul = maxcumul - mincumul + np.arange(len(maxcumul)) + 1
    cumul = cumul.reshape(len(cumul), 1, 1, 1)
    newlbld = np.where(ser > 0, cumul + ser, 0)
    if verbosity:
        for item in newlbld:
            print(item[item > 0].min(), item.max())
    # newlbld = np.array([np.where(item > 0, item + i, 0) for i, item in zip(maxcumul, ser)])
    print('Total number of labels: {}'.format(newlbld.max()))
    if dim_changed:
        newlbld = newlbld[0]
    return newlbld

def hlabel(binary, verbosity = False):
    """ Abbreviation of the hierarchical_label function """
    return hierarchical_label(binary, verbosity)


def label_with_stats(lbld, prop = 'area', img = None, extra_prop = None, return_tables = False):
    """ A function to re-label a 3D or 4D label image with a variety of object properties obtained by
        using a combination of scikit-image's regionprops and pandas. 
        
        The object properties can be either regionprops' default properties or any list of custom functions
        plugged in separately.
        
        Parameters:
        ----------
        lbld: array 
            A 3D or 4D array with labels representing objects.
        prop: str
            Name of the property to be computed per label. This represents one of the default properties 
            from the 'regionprops_table' from scikit-image. 
        img: array
            A second image with the same shape as the 'lbld'. If not None, calculation of the property is 
            done by masking this image with the label masks from the 'lbld'. 
        extra_prop: iterable of functions
            A function returning the property to be computed per label. This function will be applied per
            labeled region and the returned value will be used to relabel that region. If not None, overrules 'prop'.
        return_tables: bool
            If True, the statistical table for each volume of the 4D (or a single 3D) data will be returned in
            addition to the relabeled image. If False, only the relabeled image will be returned.
        
        Returns:
        --------
        newim: array
            A 3D or 4D array with the same shape as 'lbld'. 
        dfs: list
            A list of pandas DataFrames, containing calculated statistical properties for each label in each volume.
            Returned only if 'return_tables' is True.
        """
    assert lbld.ndim > 2
    dim_changed = 0
    if img is None:
        img = resetformat(lbld > 0) * 1.
    else:
        assert lbld.ndim == img.ndim
    if lbld.ndim < 4:
        dim_changed = 1
        lbld = lbld.reshape(1, *lbld.shape)
        img = img.reshape(1, *img.shape)
    prop = 'area' if prop is None else prop
    # get a regionprops_table dictionary for each volume
    dicto = [regionprops_table(lbld_, img_, properties = ('label', prop), extra_properties = extra_prop) 
                      for lbld_, img_ in zip(lbld, img)]    
    # convert each dictionary into a dataframe for convenience and collect them in a list
    dfs = [pd.DataFrame(dicto_) for dicto_ in dicto]
    newim = np.zeros_like(lbld).astype(float)
    for i, (vol, newvol, df) in enumerate(zip(lbld, newim, dfs)):
        for ind in df.index:
            lbl = df.loc[ind, 'label']
            propval = df.loc[ind, prop]
            newim[i][vol == lbl] = propval
            if extra_prop is not None:
                expropval = df.iloc[ind, 2]
                newim[i][vol == lbl] = expropval
    if dim_changed:
        newim = newim[0]
    if return_tables:
        return newim, dfs
    else:
        return newim


def get_largest_object(binary, per_volume = False): 
    """ Gets the object with largest volume in a 3D or 4D numpy array. 
    
        Parameters:
        -----------
        binary: array of bool (or 1s and 0s)
            a 3D numpy array with binary object(s) in it.
        per_volume: bool
            If False, the largest object is calculated from the pool of all labels together.
            If True, the largest object is calculated from the labels for each volume in the 4D.
        
        Returns:
        --------
        largest: array 
            Array with the same shape as 'binary
        """
    cp_array(binary)
    lbld = hierarchical_label(binary)
    sizelbld = label_with_stats(lbld, 'area')
    if per_volume:
        largest = np.array([item == item.max() for item in sizelbld])
    else:
        largest = sizelbld == sizelbld.max()
    return largest



def get_centroids(lblim):
    """ lblim is a 3D or 4D image with labeled objects. The function finds the centroid of each 
        object. 
        
        Parameters: 
        -----------
        lblim: array 
            A 3D or 4D array with labels representing objects. 
        
        Returns:
        --------
        cenvol: array
            Array with same shape as 'lblim' and with centroids of the objects. 
            Centroid of each object is assigned the label of the
            corresponding object from 'lblim'.
        """
    cp_array(lblim)
    dim_changed = 0
    if lblim.ndim == 3:
        dim_changed = 1
        lblim = lblim.reshape(1, *lblim.shape)
    table = [(i, regionprops_table(lbld, properties = ('label', 'centroid'))) 
             for i, lbld in enumerate(lblim.astype(int))]
    dfs = []
    for i, t in table:
        df = pd.DataFrame(t)
        t = [i] * len(df)
        df['centroid-t'] = t
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index = True)
    pts = np.around(np.array(dfs.iloc[:, 1:])).astype(int)
    pts = np.hstack((pts[:, 3:], pts[:, :3]))
    lbls = dfs['label'].to_numpy()
    cenvol = np.zeros_like(lblim)
    cenvol[tuple(pts.T)] = lbls
    if dim_changed:
        cenvol = cenvol[0]
    return cenvol
    
    
    

##############################################################################################################
##############################################################################################################