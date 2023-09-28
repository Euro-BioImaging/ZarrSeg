# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:42:12 2020

@author: bugra
"""


""" 
Here I implemented a series of local adaptive thresholding algorithms. Most of the implemented methods
were originally developed in the context of text/document segmentation. However, with a little
modification and/or extra work, they are well suitable for bioimage segmentation, too. 
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import dtype_limits

# local imports
from ZarrSeg.bin.transforms.photometric.local_filtering import statistical as st
from ZarrSeg.bin.transforms.photometric.thresholding import global_threshold as gt



### To be added: Su 2010, Gatos 2005, ISauvola 2016

def niblack(img, shape, k = -0.2, return_thresholded = True):
    """ The well-known niblack local threshold algorithm based on:
        "Niblack, W (1986), An introduction to Digital Image Processing, Prentice-Hall." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher local thresholds. Typically
            between -0.5 and +0.5.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    thresholds = mean + k * stdev
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def sauvola(img, shape, k, return_thresholded = True):
    """ The well-known sauvola local threshold algorithm based on:
        "J. Sauvola and M. Pietikainen, “Adaptive document image binarization,”
        Pattern Recognition 33(2), pp. 225-236, 2000." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower local thresholds. Typically
            between -0.2 and +0.2.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """
    # T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))   
    imin, imax = dtype_limits(img, False)
    r = 0.5 * (imax - imin)
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    # r = np.max(stdev) # another approach for r
    thresholds = mean * (1. - k * (1. - (stdev / r)))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def wolf(img, shape, k, return_thresholded = True):
    """ Wolf's local threshold algorithm based on:
        "C. Wolf, J-M. Jolion, “Extraction and Recognition of Artificial 
        Text in Multimedia Documents”, Pattern Analysis
        and Applications, 6(4):309-326, (2003)." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher thresholds. Typically
            between 0 and +0.03.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """
    imin, imax = dtype_limits(img, False)
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    minval = np.min(img)
    r = np.max(stdev)
    thresholds = ((1. - k) * mean + (k * stdev) / (r * (mean - minval)) + k * minval)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds


def singh(img, shape, k, return_thresholded = True):
    """ Singh's local threshold algorithm based on:
        "Singh OI, Sinam T, James O et al (2012), 
        “Local contrast and mean based thresholding technique in image 
        binarization.” Int J Comput Appl 51(6):4–10." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher thresholds. Typically
            between 0.25 and +0.5.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """
    mm = st.local_minmax_statistics(img, shape)    
    contrast = mm.contrast
    mean = st.mean_filter(img, shape)
    thresholds = k * (mean + contrast * (1 - img))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def trsingh(img, shape, k, return_thresholded = True):
    """ Singh's local threshold algorithm based on:
        "Singh TR, Roy S, Singh OI et al (2011) “A new local adaptive thresholding 
        technique in binarization.” Int J Comput Sci Issues 8(6):271–277" 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between -0.1 and +0.1.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """    
    mean = st.mean_filter(img, shape)
    diff = img - mean
    thresholds = mean * (1 + k * ((diff / (1-diff)) - 1))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def wan(img, shape, k = 0.33, return_thresholded = True):
    """ Wan's local threshold algorithm based on:
        "W. A. Mustafa and M. M. M. A. Kader. (2018). “Binarization of Document Image Using Optimum
        Threshold Modification.” J. Phys. Conf. Ser., 1019 (012022), pp. 1–8." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between 0.25 and +0.45.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """        
    imin, imax = dtype_limits(img, False)
    r = 0.5 * (imax - imin)
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    # r = np.max(stdev) # alternative way to r
    fp = st._get_footprint(img, shape)
    maximum = ndi.maximum_filter(img, footprint = fp)
    maxmean = 0.5 * (maximum + mean)
    thresholds = maxmean * (1. - k * (1. - (stdev / r)))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds


def nick(img, shape, k, return_thresholded = True):
    """ NICK local threshold algorithm based on:
        "Khurshid, K., Siddiqi, I., Faure, C., & Vincent, N.
        (2009, January). “Comparison of Niblack inspired Binarization methods for
        ancient documents.” In IS&T/SPIE Electronic Imaging (pp. 72470U-72470U).
        International Society for Optics and Photonics." 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher thresholds. Typically
            between -0.1 and +0.1.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.            
        """        

    fp = st._get_footprint(img, shape)
    numvox = fp.size
    summed = st.sum_filter(img, shape)
    sqsummed = st.sum_filter(np.square(img), shape)
    mean = summed / numvox
    thresholds = mean + k * np.sqrt((sqsummed - mean ** 2) / numvox)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def nick_v2(img, shape, k, return_thresholded = True):
    """ A variation of the NICK local threshold algorithm (Khurshid et al. 2009).
        This version is adapted from: https://www.mathworks.com/matlabcentral/fileexchange/42104-nick-local-image-thresholding
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher thresholds. Typically
            between -0.1 and +0.1.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.         
        """ 
    mean = st.mean_filter(img, shape)
    meansq = st.mean_filter(np.square(img), shape)
    variance = meansq - np.square(mean)
    thresholds = mean + k * np.sqrt(np.square(mean) + variance)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def nick_v3(img, shape, k, return_thresholded = True):
    """ My own adaptation of the NICK local threshold algorithm (Khurshid et al. 2009).
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to higher thresholds. Typically
            between 0 and +0.05.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.         
        """ 
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    thresholds = mean + k * np.sqrt(np.square(mean) + np.sqrt(stdev))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def laab(img, shape, k, return_thresholded = True): 
    """ Local adaptive transformation method based on:
        "Romen T Singh, Sudipta Roy and Kh. Manglem Singh. “Local Adaptive 
        Automatic Binarisation (LAAB).” International Journal of Computer Applications 40(6):27-30, February 2012."
        
        Note that this method is originally not a thresholding but an automatic binarisation method. 
        However, for consistency with other functions in this module, we present it here as a thresholding method.
        Therefore, it is possible to plug this method in the hybrid_bernsen
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between 0.4 and 0.6.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.         
        """ 
    if k < 0.4 or k > 0.6:
        print ("k OUTSIDE THE RANGE OF {0.4-0.6} IS NOT RECOMMENDED.")
    mean = st.mean_filter(img, shape)
    diff = img - mean
    delta = diff * (1 - mean)
    v = k * ((1 + delta) / (1 - delta))
    b = (np.abs(1 - 2 * v) - (1 - 2 * v)) / (2 * np.abs(1 - 2 * v))
    if return_thresholded:
        return b
    else:
        return (1 - b)
    
def contrast(img, shape, k = 1, return_thresholded = True):
    """ A contrast based threshold method. This method was adapted from: 
        https://github.com/manuelaguadomtz/pythreshold/blob/master/pythreshold/local_th/contrast.py
                
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between 0.8 and 1.2. Cannot be below 0 or above 2.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.         
        """ 
    assert (k < 2) & (k > 0), 'Only values in the range of (0, 2) are allowed.'
    mm = st.local_minmax_statistics(img, shape)    
    minis = mm.minis
    maxis = mm.maxis
    minshift = (img - minis) * k
    maxshift = (maxis - img) * (2 - k)
    thresholds = np.where(minshift <= maxshift, img.max(), 0)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds



#################################### Methods that do not need the  parameter k ##########################################
#########################################################################################################################


def bataineh(img, shape, return_thresholded = True):
    """ Bataineh local threshold algorithm based on 
        "Bilal Bataineh, Siti Norul Huda Sheikh Abdullah, Khairuddin Omar
        “An adaptive local binarization method for document images based 
        on a novel thresholding method and dynamic windows”
        October 2011, Pattern Recognition Letters 32(14):1805-1813"
        
        Note that this method does not require a threshold scaling constant k.
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.         
        """ 
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    meansq = np.square(mean)
    stdev = bls.stdev
    meang = np.mean(img)
    dev_adapt = ((stdev - np.min(stdev)) / (np.max(stdev) - np.min(stdev))) * np.max(img)
    thresholds = mean - ((meansq * stdev) / ((meang + stdev) * (dev_adapt + stdev)))
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def rais(img, shape, return_thresholded = True, l = -0.3):
    """ Thresholding method based on the paper: N. B. Rais, M. S. Hanif and I. A. Taj, 
        “Adaptive thresholding technique for document image analysis” 8th International 
        Multitopic Conference, 2004. Proceedings of INMIC 2004., 2004, pp. 61-66, doi: 10.1109/INMIC.2004.1492847.
        
        This method is a modification of Niblack method such that it identifies
        the factor k automatically. Note that the parameter l is strictly -0.3 in the 
        original paper and this value should normally not be altered. 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
        l: float
            A scalar used in the calculation of k. This scalar's default value
            is -0.3 and this value should normally not be changed. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.   
        """
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    prodl = mean * stdev
    mean_g = np.full(mean.shape, np.mean(img))
    stdev_g = np.full(mean.shape, np.std(img))
    prodg = mean_g * stdev_g            
    prodmax = np.max([prodl, prodg], axis = 0)
    prodmax = st._block_zeroes(prodmax)
    k = l * (prodg - prodl) / prodmax
    # k = (mean_g - mean) / max_mean
    thresholds = mean + k * stdev
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds


############################################## Bernsen-based methods ####################################################
#########################################################################################################################


def bernsen(img, shape, min_contrast = 0.1,
                       return_thresholded = True):
    """ Thresholding method based on the paper: "Bernsen, J (1986), “Dynamic Thresholding of Grey-Level Images”,
        Proc. of the 8th Int. Conf. on Pattern Recognition"
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        min_contrast: float
            The contrast threshold that is used to decide whether the voxel in
            a local neighbourhood should be subjected to local or global threshold.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.   
        """
    mm = st.local_minmax_statistics(img, shape)    
    contrast = mm.contrast
    thresholds = mm.midpoint
    threshold_g = gt.otsu(img)
    thresholds = np.where(contrast > min_contrast, thresholds, threshold_g)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def _generic_contrast_mask(img, shape, min_contrast = 0.1):
    """ This function calculates a contrast mask just as done in the bernsen
        thresholding method. This mask is used in a generic way to distinguish the
        voxels that should be segmented via local thresholds from those that
        should be segmented via a global threshold. This principle is the basis of
        the 'hybrid_bernsen' function.
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        min_contrast: float
            The contrast threshold that is used to decide whether the voxel in
            a local neighbourhood should be subjected to local or global threshold.
            
        Returns:
        --------
        contrast_mask: array of bool
            Numpy array with the same shape as 'img'.
        """
    mm = st.local_minmax_statistics(img, shape)    
    contrast = mm.contrast
    return contrast < min_contrast


def hybrid_bernsen(img, shape, method, 
                   k, min_contrast = 0.1,
                   return_thresholded = True):
    """ This function combines the local contrast masking feature of the bernsen 
        method with the other local threshold techniques implemented in this module.
        The end result is a highly flexible local threshold calculator, which applies 
        the desired local threshold method exclusively to the high-contrast zones in the image.
        Thus, the binary noise arising from the missegmentations can be effectively avoided.
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        method: function
            Any of the local threshold functions in this module, except 'phansalkar' and 'feng',
            can be passed. 
        k: float
            A factor to manually tune the local thresholds. The optimal selection of k value depends on 
            the local threshold function passed to 'method'.
        min_contrast: float
            The contrast threshold that is used to decide whether the voxel in
            a local neighbourhood should be subjected to local or global threshold.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False. """
    mask = _generic_contrast_mask(img, shape, min_contrast)
    if (method.__name__ == 'rais') | (method.__name__ == 'bataineh'):
        thresholds = method(img, shape, return_thresholded = False)        
    else:
        thresholds = method(img, shape, k, return_thresholded = False)
    threshold_g = gt.otsu(img)
    thresholds[mask] = threshold_g
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds



################################# Methods requiring large number of parameters ##########################################
#########################################################################################################################


def phansalkar(img, shape, k, p = 2, q = 10, return_thresholded = True):
    """ Phansalkar local thresholding algorithm based on: 
        N. Phansalkar, S. More, A. Sabale, and M. Joshi, 
        “Adaptive local thresholding for detection of nuclei in diversity stained cytology images,” 
        in 2011 International Conference on Communications and Signal Processing, 2011, pp. 218-220: IEEE.
        In this method, the higher k and q and lower p segments more. 
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between 0 and +0.1.
        p: float
            A factor to manually tune the local thresholds.
            Higher p leads to higher thresholds. Typically
            between 2 and 5.            
        q: float
            A factor to manually tune the local thresholds.
            Higher q leads to lower thresholds. Typically
            between 5 and 15.                    
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.  
        """
    bls = st.basic_local_statistics(img, shape)
    mean = bls.mean
    stdev = bls.stdev
    r = np.max(stdev)
    thresholds = mean * (1. + p * np.exp(-q * mean) + k * ((stdev / r) - 1)) 
    if return_thresholded:
        return (img > thresholds)
    else:
        return thresholds


def feng(img, small_shape = 5, big_shape = 7, alpha = 0.18, 
                    k1 = 0.66, k2 = 0.15, gamma = 2, return_thresholded = True):
    """ Feng local thresholding algorithm based on: Meng-Ling Feng and Yap-Peng Tan, 
        "Contrast adaptive binarization of low quality document images", 
        IEICE Electron. Express, Vol. 1, No. 16, pp.501-506, (2004)).
    
        Threshold formula is like: term0 + term1 + term2,
        where term0 is controlled by alpha, term1 by alpha1 and term2 by alpha2.
        
        Parameters:
        -----------
        small_shape: scalar or tuple or list or array
            The parameter for small window dimensions
        big_shape: scalar or tuple or list or array
            The parameter for large window dimensions        
        alpha: float 
            Factor that linearly controls term0. Typically between 0.1 and 0.2. 
            Higher alpha leads to lower threshold.
        k1: float
            Factor that linearly controls alpha1, hence term1. Typically between 0.15 and 0.25. 
            Higher k1 leads to higher threshold.
        k2: float
            Factor that linearly controls alpha2, hence term2. Typically between 0.01 and 0.05. 
            Higher k2 leads to higher threshold.
        gamma: float
            Factor that nonlinearly controls alpha1 and alpha2, hence term1 and term2.
            Typically 2.     

        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.  
        """
    bls_small = st.basic_local_statistics(img, small_shape)
    mean_small = bls_small.mean
    stdev_small = bls_small.stdev
    fp = st._get_footprint(img, small_shape)
    min_small = ndi.minimum_filter(img, footprint = fp)
    ####
    bls_big = st.basic_local_statistics(img, big_shape)
    stdev_big = bls_big.stdev
    #####
    alpha2 = k1 * (stdev_small / stdev_big) ** gamma
    alpha3 = k2 * (stdev_small / stdev_big) ** gamma
    #####
    term0 = (1 - alpha) * mean_small
    term1 = alpha2 * (stdev_small / stdev_big) * (mean_small - min_small) 
    term2 = alpha3 * min_small
    thresholds = np.nan_to_num(term0 + term1 + term2)
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds


def phansalkar_bernsen(img, shape, k = 0.05, p = 10, q = 30, min_contrast = 0.1, 
                    return_thresholded = True):
    """ Hybrid version of the Phansalkar method. In this function, the
        Phansalkar local thresholding is applied exclusively to high-contrast
        voxels selected by "min_contrast" parameter. The low-contrast voxels
        that are not included in this mask are segmented via Otsu thresholding.
        
        Parameters:
        -----------
        shape: scalar or tuple or list or array
            The parameter for window dimensions.
        k: float
            A factor to manually tune the local thresholds.
            Higher k leads to lower thresholds. Typically
            between 0 and +0.1.
        p: float
            A factor to manually tune the local thresholds.
            Higher p leads to higher thresholds. Typically
            between 5 and 25.            
        q: float
            A factor to manually tune the local thresholds.
            Higher q leads to lower thresholds. Typically
            between 10 and 50.            
        min_contrast: float
            The contrast threshold that is used to decide whether the voxel in
            a local neighbourhood should be subjected to local or global threshold.
        return_thresholded: bool
            Specifies whether the output is the threshold values 
            or the thresholded image. 
            
        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.  
        """
    mask = _generic_contrast_mask(img, shape, min_contrast)
    thresholds = phansalkar(img, shape, k, p, q, False)    
    threshold_g = gt.otsu(img)
    thresholds[mask] = threshold_g
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds

def feng_bernsen(img, small_shape = 3, big_shape = 5, alpha = 0.12, 
                    k1 = 0.25, k2 = 0.04, gamma = 2, min_contrast = 0.1, 
                    return_thresholded = True):
    """ Hybrid version of the Feng method. In this function, the
        Feng local thresholding is applied exclusively to high-contrast
        voxels selected by "min_contrast" parameter. The low-contrast voxels
        that are not included in this mask are segmented via Otsu thresholding.
    
        Feng threshold formula is like: term0 + term1 + term2,
        where term0 is controlled by alpha, term1 by alpha1 and term2 by alpha2.
        
        Parameters:
        -----------
        small_shape: scalar or tuple or list or array
            The parameter for small window dimensions
        big_shape: scalar or tuple or list or array
            The parameter for large window dimensions        
        alpha: float 
            Factor that linearly controls term0. Typically between 0.1 and 0.2. 
            Higher alpha leads to lower threshold.
        k1: float
            Factor that linearly controls alpha1, hence term1. Typically between 0.15 and 0.25. 
            Higher k1 leads to higher threshold.
        k2: float
            Factor that linearly controls alpha2, hence term2. Typically between 0.01 and 0.05. 
            Higher k2 leads to higher threshold.
        gamma: float
            Factor that nonlinearly controls alpha1 and alpha2, hence term1 and term2.
            Typically 2.     

        Returns:
        --------
        thresholded: array of bool
            Array with the same shape as the 'img'. Returned
            if return_threshold is True.
        thresholds: array of float
            Array with the same shape as the 'img'. Returned
            if return_threshold is False.  
        """
    mask = _generic_contrast_mask(img, small_shape, min_contrast)
    thresholds = feng(img, small_shape, big_shape, alpha, k1, k2, gamma, False)    
    threshold_g = gt.otsu(img)
    thresholds[mask] = threshold_g
    if return_thresholded:
        return(img > thresholds)
    else:
        return thresholds



