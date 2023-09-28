# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:42:12 2020

@author: bugra
"""

import numpy as np, dask as da

def manual_threshold(im, low, high = None):
    """ Gets a binary mask for regions above low and below high.
        
        Parameters:
        -----------
        im: array
            Image to be thresholded.
        
        Returns:
        --------
        binary: array of bool
            The thresholded binary image. """
    if low is not None:
        mask0 = im > low
    else:
        mask0 = 1
    if high is not None:
        mask1 = im < high
    else:
        mask1 = 1
    return mask0 * mask1


def isodata(im, bincount = 2000, return_thresholded = False):
    """ Applies the isodata thresholding algorithm based on 
        "Ridler, TW & Calvard, S (1978), “Picture thresholding using an iterative selection method” 
        IEEE Transactions on Systems, Man and Cybernetics 8: 630-632, DOI:10.1109/TSMC.1978.4310039".
        
        This algorithm finds a threshold t, which satisfies the criterion:
            t = (mean_lower(t) + (mean_higher(t))) / 2,
            where mean_lower is the mean of all intensities below t and
            mean_higher is the mean of all intensities above t.
            
        Parameters:
        -----------
        im: array
            n-dimensional intensity array to be subjected to isodata threshold calculation.
        bincount: int
            a scalar that specifies the count of bins used for the histogram calculation.
        return_thresholded: bool
            specifies if the threshold should be applied to the 'img' or directly returned.
            
        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        
        """    
    img = da.array.from_zarr(im)
    hist, vals = da.array.histogram(img, bins=bincount, range=[img.min(), img.max()])
    binmids = 0.5 * (vals[1:] + vals[:-1]) # get the midpoints of the bins
    countsums0 = np.cumsum(hist) # cumulative sum of the voxel counts in the bins, starting from the minimum
    countsums1 = np.cumsum(hist[::-1])[::-1] # cumulative sum of the voxel counts in the bins, starting from the maximum
    binsums = binmids * hist # sum of intensities within each bin
    valuesums0 = np.cumsum(binsums) # cumulative sum of intensities, starting from the minimum
    valuesums1 = np.cumsum(binsums[::-1])[::-1] # cumulative sum of intensities, starting from the maximum
    cummeans0 = valuesums0 / countsums0 # cumulative mean of intensities, starting from the minimum
    cummeans1 = valuesums1 / countsums1 # cumulative mean of intensities, starting from the maximum
    cummids = 0.5 * (cummeans0 + cummeans1) # average of the class mean values
    objective = np.abs(binmids - cummids) # the closest distance between the bin mid-values and the average of the class means.
    argt = np.argmin(objective) # index corresponding to the minimum of the objective 
    t = binmids[argt]
    if return_thresholded:
        return img > t
    else:
        return t

def otsu(im, bincount=1024, return_thresholded=False):
    """ Applies the Otsu thresholding algorithm based on: "Nobuyuki Otsu (1979).
            “A threshold selection method from gray-level histograms”.
            IEEE Trans. Sys. Man. Cyber. 9 (1): 62–66. doi:10.1109/TSMC.1979.4310076".

        This algorithm finds a threshold t, which maximises the equation:
            (var_higher(t) - var_lower(t)),
            where var_lower is the weighted variance of all intensities below t and
            var_higher is the weighted variance of all intensities above t.

        Parameters:
        -----------
        im: dask.array
            n-dimensional intensity array to be subjected to Otsu threshold calculation.
        bincount: int
            a scalar that specifies the count of bins used for the histogram calculation.
        return_thresholded: bool
            specifies if the threshold should be applied to the 'img' or directly returned.

        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'.
            Returned only if return_thresholded is True.

        """
    img = da.array.from_zarr(im)
    hist, vals = da.array.histogram(img, bins=bincount, range=[img.min(), img.max()])
    binmids = 0.5 * (vals[1:] + vals[:-1])  # get the midpoints of the bins
    countsums0 = np.cumsum(hist)  # cumulative sum of the voxel counts in the bins, starting from the minimum
    countsums1 = np.cumsum(hist[::-1])[
                 ::-1]  # cumulative sum of the voxel counts in the bins, starting from the maximum
    binsums = binmids * hist  # sum of intensities within each bin
    valuesums0 = np.cumsum(binsums)  # cumulative sum of intensities, starting from the minimum
    valuesums1 = np.cumsum(binsums[::-1])[::-1]  # cumulative sum of intensities, starting from the maximum
    cummeans0 = valuesums0 / countsums0  # cumulative mean of intensities, starting from the minimum
    cummeans1 = valuesums1 / countsums1  # cumulative mean of intensities, starting from the maximum
    objective = countsums0[:-1] * countsums1[1:] * (cummeans0[:-1] - cummeans1[1:]) ** 2  # interclass variance equation. Frameshift ensures correct bin match
    argt = np.argmax(objective)  # index of the maximum interclass variation
    t = binmids[argt]  # threshold value corresponding to maximum interclass variation
    if return_thresholded:
        return img > t
    else:
        return t


# def otsu(img, bincount = 1024, return_thresholded = False):
#     """ Applies the Otsu thresholding algorithm based on: "Nobuyuki Otsu (1979).
#             “A threshold selection method from gray-level histograms”.
#             IEEE Trans. Sys. Man. Cyber. 9 (1): 62–66. doi:10.1109/TSMC.1979.4310076".
#
#         This algorithm finds a threshold t, which maximises the equation:
#             (var_higher(t) - var_lower(t)),
#             where var_lower is the weighted variance of all intensities below t and
#             var_higher is the weighted variance of all intensities above t.
#
#         Parameters:
#         -----------
#         img: array
#             n-dimensional intensity array to be subjected to isodata threshold calculation.
#         bincount: int
#             a scalar that specifies the count of bins used for the histogram calculation.
#         return_thresholded: bool
#             specifies if the threshold should be applied to the 'img' or directly returned.
#
#         Returns:
#         --------
#         t: float
#             scalar float indicating the threshold value.
#         thresholded: array of 1s and 0s
#             n-dimensional binary mask produced by applying the threshold 't' to the 'img'.
#             Returned only if return_thresholded is True.
#
#         """
#     hist, vals = np.histogram(img, bincount) # calculate histogram
#     binmids = 0.5 * (vals[1:] + vals[:-1]) # get the midpoints of the bins
#     countsums0 = np.cumsum(hist) # cumulative sum of the voxel counts in the bins, starting from the minimum
#     countsums1 = np.cumsum(hist[::-1])[::-1] # cumulative sum of the voxel counts in the bins, starting from the maximum
#     binsums = binmids * hist # sum of intensities within each bin
#     valuesums0 = np.cumsum(binsums) # cumulative sum of intensities, starting from the minimum
#     valuesums1 = np.cumsum(binsums[::-1])[::-1] # cumulative sum of intensities, starting from the maximum
#     cummeans0 = valuesums0 / countsums0 # cumulative mean of intensities, starting from the minimum
#     cummeans1 = valuesums1 / countsums1 # cumulative mean of intensities, starting from the maximum
#     objective = countsums0[:-1] * countsums1[1:] * (cummeans0[:-1] - cummeans1[1:]) ** 2 # interclass variance equation. Frameshift ensures correct bin match
#     argt = np.argmax(objective) # index of the maximum interclass variation
#     t = binmids[argt] # threshold value corresponding to maximum interclass variation
#     if return_thresholded:
#         return img > t
#     else:
#         return t
    

def yen(im, bincount = 1024, return_thresholded = False):
    """ Applies the Yen thresholding algorithm based on 
        "Yen J.C., Chang F.J., and Chang S. (1995) “A New Criterion for Automatic Multilevel Thresholding” 
            IEEE Trans. on Image Processing, 4(3): 370-378. DOI:10.1109/83.366472".
            
        
        This function was adapted from the scikit-image library:
        https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py#L381-L439
            

        Parameters:
        -----------
        im: array
            n-dimensional intensity array to be subjected to isodata threshold calculation.
        bincount: int
            a scalar that specifies the count of bins used for the histogram calculation.
        return_thresholded: bool
            specifies if the threshold should be applied to the 'img' or directly returned.
            
        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        
        """
    img = da.array.from_zarr(im)
    hist, vals = da.array.histogram(img, bins=bincount, range=[img.min(), img.max()])
    binmids = 0.5 * (vals[1:] + vals[:-1]) # get the midpoints of the bins
    probs0 = hist / hist.sum() # get probability distribution from histogram counts
    probs1 = probs0[::-1]
    cumprobs0 = np.cumsum(probs0) 
    cumprobs0_sq = np.cumsum(probs0 ** 2)
    cumprobs1_sq = np.cumsum(probs1 ** 2)[::-1]
    objective = np.log(((cumprobs0[:-1] * (1. - cumprobs0[:-1])) ** 2) / (cumprobs0_sq[:-1] * cumprobs1_sq[1:]))
    argt = np.argmax(objective)
    t = binmids[argt]
    if return_thresholded:
        return img > t
    else:
        return t
       

def ridler(im, tol = 0.01, return_thresholded = False, random_start = False, verbous = False):
    """ Applies the Ridler & Calvard's iterative thresholding algorithm. 
        
        Reference: "Ridler, TW & Calvard, S (1978), “Picture thresholding using an iterative selection method” 
                    IEEE Transactions on Systems, Man and Cybernetics 8: 630-632, DOI:10.1109/TSMC.1978.4310039".

        The algorithm iterates the threshold t such that t is always the midpoint of 
        the means of the foreground and background voxels. This is a modified 
        version of the 'isodata' function.
    
        Parameters:
        -----------
        img: array
            N-dimensional intensity array to be subjected to ridler threshold calculation.
        tol: float
            The tolerance threshold for the loss. If loss is below 'tol', the iteration is terminated.
        return_thresholded: bool
            Specifies if the threshold should be applied to the 'img' or directly returned.
        random_start: bool
            Specifies if the threshold is randomly initialised. If False, the initial threshold
            is the midpoint of the minimum nonzero value and the maximum value.
        verbous: bool
            If True, the loss is printed.   
            
        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        
        """
    img = da.array.from_zarr(im)
    vals = img.ravel()
    if random_start:
        t_init = np.random.choice(vals)
    else:
        t_init = 0.5 * (vals[vals > 0].min() + vals.max())
    t = t0 = t_init; loss = np.inf
    while loss > tol:
        mean0 = vals[vals <= t].mean()
        mean1 = vals[vals > t].mean()
        t = 0.5 * (mean0 + mean1)
        loss = np.abs(t - t0)
        t0 = t
    if verbous:
        print(loss)
    if return_thresholded:
        return img > t
    else:
        return t            


def ridler_wmean(im, bincount = 1024, tol = 0.01, return_thresholded = False, random_start = False, verbous = False):
    """ Applies a modification of the Ridler & Calvard's iterative thresholding algorithm. Instead of means,
        this algorithm calculates weighted means for each class, using image histogram.

        Reference: "Ridler, TW & Calvard, S (1978), “Picture thresholding using an iterative selection method” 
                    IEEE Transactions on Systems, Man and Cybernetics 8: 630-632, DOI:10.1109/TSMC.1978.4310039".

        Parameters:
        -----------
        im: array
            N-dimensional intensity array to be subjected to isodata threshold calculation.
        bincount: int
            A scalar that specifies the count of bins used for the histogram calculation.
        tol: float
            The tolerance threshold for the loss. If loss is below 'tol', the iterations are terminated.
        return_thresholded: bool
            Specifies if the threshold should be applied to the 'img' or directly returned.
        random_start: bool
            Specifies if the threshold is randomly initialised. If False, the initial threshold
            is the midpoint of the minimum nonzero value and the maximum value.
        verbous: bool
            If True, the loss is printed.            

        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        """
    img = da.array.from_zarr(im)
    hist, vals = da.array.histogram(img, bins=bincount, range=[img.min(), img.max()])
    binmids = 0.5 * (vals[1:] + vals[:-1]) # get the midpoints of the bins
    if random_start:
        t_init = np.random.choice(vals)
    else:
        t_init = 0.5 * (vals[vals > 0].min() + vals.max())
    t = t0 = t_init; loss = np.inf
    while loss > tol:
        mask0 = binmids <= t
        mask1 = binmids > t
        count0 = hist[mask0]
        count1 = hist[mask1]
        mean0 = np.sum(binmids[mask0] * count0) / np.sum(count0) 
        mean1 = np.sum(binmids[mask1] * count1) / np.sum(count1) 
        t = 0.5 * (mean1 + mean0) 
        loss = np.abs(t - t0)
        t0 = t
    if verbous:
        print(loss)
    if return_thresholded:
        return img > t
    else:
        return t    
    

def ridler_median(im, tol = 0.01, return_thresholded = False, random_start = False, verbous = False):
    """ Applies a modification of the Riddler's iterative thresholding algorithm. Instead of means,
        this algorithm calculates medians for each class.

        Reference: "Ridler, TW & Calvard, S (1978), “Picture thresholding using an iterative selection method” 
                    IEEE Transactions on Systems, Man and Cybernetics 8: 630-632, DOI:10.1109/TSMC.1978.4310039".
    
        Parameters:
        -----------
        im: array
            N-dimensional intensity array to be subjected to ridler threshold calculation.
        tol: float
            The tolerance threshold for the loss. If loss is below 'tol', the iterations are terminated.
        return_thresholded: bool
            Specifies if the threshold should be applied to the 'img' or directly returned.
        random_start: bool
            Specifies if the threshold is randomly initialised. If False, the initial threshold
            is the midpoint of the minimum nonzero value and the maximum value.
        verbous: bool
            If True, the loss is printed.   
            
        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        """
    img = da.array.from_zarr(im)
    vals = img.ravel()
    if random_start:
        t_init = np.random.choice(vals)
    else:
        t_init = 0.5 * (vals[vals > 0].min() + vals.max())
    t = t0 = t_init; loss = np.inf
    while loss > tol:
        med0 = np.median(vals[vals <= t])
        med1 = np.median(vals[vals > t])
        t = 0.5 * (med0 + med1)
        loss = np.abs(t - t0)
        t0 = t
    if verbous:
        print(loss)
    if return_thresholded:
        return img > t
    else:
        return t            
                    

def li(im, bincount = 1024, tol = 0.01, return_thresholded = False, random_start = False, verbous = False):
    """ Applies Li's thresholding method based on: 
    
        "Li C.H. and Lee C.K. (1993) “Minimum Cross Entropy Thresholding” Pattern Recognition, 
        26(4): 617-625 DOI:10.1016/0031-3203(93)90115-D""
        
        and 
        
        "Li C.H. and Tam P.K.S. (1998) “An Iterative Algorithm for Minimum Cross Entropy Thresholding” 
        Pattern Recognition Letters, 18(8): 771-776 DOI:10.1016/S0167-8655(98)00057-9""
    
        This algorithm is based on minimum cross entropy between foreground and background. 
    
        Parameters:
        -----------
        im: array
            N-dimensional intensity array to be subjected to Li threshold calculation.
        bincount: int
            A scalar that specifies the count of bins used for the histogram calculation.
        tol: float
            The tolerance threshold for the loss. If loss is below 'tol', the iterations are terminated.
        return_thresholded: bool
            Specifies if the threshold should be applied to the 'img' or directly returned.
        random_start: bool
            Specifies if the threshold is randomly initialised. If False, the initial threshold
            is the midpoint of the minimum nonzero value and the maximum value.
        verbous: bool
            If True, the loss is printed.   
            
        Returns:
        --------
        t: float
            scalar float indicating the threshold value.
        thresholded: array of 1s and 0s
            n-dimensional binary mask produced by applying the threshold 't' to the 'img'. 
            Returned only if return_thresholded is True.
        """      
    img = da.array.from_zarr(im)
    hist, vals = da.array.histogram(img, bins=bincount, range=[img.min(), img.max()])
    binmids = 0.5 * (vals[1:] + vals[:-1]) # get the midpoints of the bins
    if random_start:
        t_init = np.random.choice(vals)
    else:
        t_init = 0.5 * (vals[vals > 0].min() + vals.max())
    t = t0 = t_init; loss = np.inf
    while loss > tol:
        mask0 = binmids <= t
        mask1 = binmids > t
        count0 = hist[mask0]
        count1 = hist[mask1]
        mean0 = np.sum(binmids[mask0] * count0) / np.sum(count0) 
        mean1 = np.sum(binmids[mask1] * count1) / np.sum(count1) 
        t = (mean1 - mean0) / (np.log(mean1) - np.log(mean0))
        loss = np.abs(t - t0)
        t0 = t
    if verbous:
        print(loss)
    if return_thresholded:
        return img > t
    else:
        return t         




            
            
            
            
            
            

    
    