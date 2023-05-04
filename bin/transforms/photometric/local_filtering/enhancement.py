# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 02:02:49 2021

@author: bugra
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

# local imports
from . import statistical as st
from ZarrSeg.bin.utils import convenience as cnv
from . import derivatives as der
from ..thresholding import local_threshold as lt

def generate_gaussian_kernel(kernel_shape = (20, 20, 20), kernel_weights = (1, 1, 1)):
    """ Creates Gaussian kernel with ndimensional shapes. This is a robust function, which sits the
        kernel at the center of the array specified by the 'kernel_shape'. This means the center parameter
        of the Gaussian function is automatically determined. The kernel is also normalised, which means the 
        height parameter of the Gaussian is also automatically assigned. The user only needs to specify the 
        kernel_weights which are proportional to the width of the Gaussian. 
        
        Parameters:
        -----------
        kernel_shape: iterable 
            Dimensions of the kernel. For 3D Gaussian, an iterable of size 3 should be passed.
            
        kernel_weights: iterable
            An iterable of size ndim, where ndim is the number of dimensions of the Gaussian function. 
            The kernel weights are proportional to the widths (variances) for the corresponding 
            dimensions of the Gaussian. 
            
        Returns: 
        --------
        kernel: array
            Gaussian kernel with the shape specified with kernel_shape. Kernel is normalised so the sum
            of its values equals 1.             
        """
    # g = a * exp(-(x - b) ** 2 / c ** 2), Here a is curve height. b is curve peak's coordinate. c is curve width. 
    nh = np.array(kernel_shape)
    k = np.array(kernel_weights) 
    func = lambda i, item: item[i] // 2 + 1 if item[i] % 2 > 0 else item[i] // 2 
    dimensions = [np.exp(-(np.linspace(-nh[i] // 2, func(i, nh), nh[i]) / k[i]) ** 2) for i in range(len(kernel_shape))]
    # dimensions = [np.exp(-(np.linspace(-nh[i] // 2, nh[i] // 2 + 1, nh[i]) / k[i]) ** 2) for i in range(len(kernel_shape))]
    # g = np.meshgrid(row / np.trapz(row), depth / np.trapz(depth), col / np.trapz(col))
    g = np.meshgrid(* dimensions, indexing = 'ij')
    kernel = np.prod(g, axis = 0)
    return kernel / kernel.sum()

def generate_gaussian_cylinder(kernel_shape = (20, 20, 20), kernel_weights = (1, 1, 1)):
    """ Returns a cylindrical gaussian kernel, which is aligned along the axis (0, 0, 1). """
    nh = np.array(kernel_shape)
    k = np.array(kernel_weights)     
    func = lambda i, item: item[i] // 2 + 1 if item[i] % 2 > 0 else item[i] // 2 
    dimensions = [np.exp(-(np.linspace(-nh[i] // 2, func(i, nh), nh[i]) / k[i]) ** 2) for i in range(len(kernel_shape))]
    g = np.meshgrid(* dimensions, indexing = 'ij')
    kernel = np.prod(g[:-1], axis = 0)
    return kernel / kernel.sum()

def rlu_deconvolve(image, psf, iterations = 5, kernel_type = 'gaussian', verbose = False):
    # a good psf for my images: generate_gaussian_kernel((10, 7, 7), (3, 2, 2)) 
    """ Skimage's Richardson Lucy deconvolution with slight modifications. 
    
        Parameters:
        -----------
        image: array
            An n-dimensional numpy array.
        psf: array or iterable
            Either a numpy array specifying the point spread function, or an iterable specifying the shape of the psf. 
            If directly point spread function, it must have the same number of dimensions as 'image'. 
            If an iterable, then 'kernel_type' must be specified to calculate the psf.
        iterations: int
            Number of iterations of the deconvolution
        kernel_type: str
            Either 'mean' or 'gaussian'. Conveniently calculates a point spread function.
            Ignored if psf is an array.
            
        Returns:
        --------        
        deconvolved: array
            Array with same shape as 'image'."""
    cnv.cp_array(image)
    if hasattr(psf, 'nonzero'):
        assert image.ndim == psf.ndim, 'The image and psf must have the same number of dimensions.'
    elif np.isscalar(psf):
        psf = [psf] * image.ndim
    elif hasattr(psf, 'index'):
        if kernel_type == 'mean':
            psf = np.ones(psf) / np.prod(psf)
        elif kernel_type == 'gaussian':
            w = [i // 3 for i in psf]
            psf = generate_gaussian_kernel(psf, w)
        else:
            raise TypeError('kernel_type must be either of "mean" or "gaussian"')
    image = image.astype(float)
    psf = st._block_zeroes(psf)
    im_deconv = np.full(image.shape, 0.5)
    psf_mirror = psf[::-1, ::-1, ::-1]
    conv_im = fftconvolve(im_deconv, psf_mirror, mode = 'same') 
    for i in range(iterations):
        conv_im = st._block_zeroes(conv_im)
        relative_blur = image / conv_im 
        c = fftconvolve(relative_blur, psf, mode = 'same') 
        im_deconv *= c
        conv_im = fftconvolve(im_deconv, psf_mirror, mode = 'same') 
        if verbose:
            print('iteration: {}'.format(i))
    return im_deconv

def iterative_unsharp_mask (img, k = 1.2, iterations = 1, window = 3, blur_func = ndi.gaussian_filter):
    """ An iterative sharpening function based on unsharp mask filter.
        
        Parameters:
        -----------
        img: array
            The n-dimensional numpy array representing the input image.
        k: float
            Determines the strength of the sharpening. Higher values sharpen more but may remove 
            foreground signal too. Better not to be higher than 1.
        iterations: int
            Number of iterations of the unsharp mask operation (I - g(I)) where g is the blurring function.
            It is recommended to keep 'iterations' high and the 'k' parameter low.
        window: scalar or iterable
            Specifies the local window shape for the blur_func or the 'sigma' parameter if the 'blur_func' is Gaussian. 
            This parameter is passed to the 'blur_func'
        blur_func: function
            A transformation function that blurs the image. Usually a Gaussian or mean filter.
        
        Returns:
        --------
        deblurred: array 
            Numpy array representing the sharpened image. Has the same shape as 'img'.
        
    """
    deblurred = img.astype(float)
    e = np.finfo(deblurred.dtype).eps
    for i in range(iterations):
        blur = blur_func(deblurred, window)
        deblurred = deblurred - blur * k
        deblurred[deblurred < 0] = e
        deblurred = cnv.rescale(deblurred)
    return deblurred



################################################### Blobness filters ######################################################
###########################################################################################################################
def doh_filter (img, sig = 1, min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Detects blobs based on determinants of hessian.
        
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the size of the blobs to be detected. 
            By passing an iterable of sigmas, blobs with a variety of sizes can be detected.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        blobness: array 
            Numpy array representing the output image with enhanced blobs. Has the same shape as 'img'.
        
    """
    cnv.cp_3d(img)
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig:
        blobness = der.determinant_hessian_matrix(img, i, scale, grad_type)
        cube.append(blobness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)    
    return res * (1 - cmask)  

def highlight_blobs_hessian(img, sig = 1, min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Detects blobs based on eigenvalues of the hessian using formula: l2^2 / l0.
        
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the size of the blobs to be detected. 
            By passing an iterable of sigmas, blobs with a variety of sizes can be detected.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        blobness: array 
            Numpy array representing the output image with enhanced blobs. Has the same shape as 'img'.
        
    """
    cnv.cp_3d(img)
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig:
        l0, l1, l2 = der.eigvals_hessian_matrix(img, i, scale, grad_type)
        denom = der.st._block_zeroes(l0)
        blobness = np.where((l0 < 0) & (l1 < 0) & (l2 < 0), np.square(l2) / np.abs(denom), 0)
        cube.append(blobness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)    
    return res * (1 - cmask)

def jerman_blobness(img, sig = 1, tau = 0.5, min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Jerman's blob-detection method based on the paper:
        "" 'Blob Enhancement and Visualization for Improved Intracranial Aneurysm Detection', 
            IEEE Transactions on Visualization and Computer Graphics, 22(6), p. 1705-1717 (2016) ""
            
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the size of the blobs to be detected. 
            By passing an iterable of sigmas, blobs with a variety of sizes can be detected.
        tau: float
            Parameter to adjust the intensity of the blobness response. Lower tau leads to more intense response.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        blobness: array 
            Numpy array representing the output image with enhanced blobs. Has the same shape as 'img'.
        
    """
    cnv.cp_3d(img)
    assert img.max() == 1., 'Input image is not normalised. First normalise it between 0 and 1.'
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig:    
        eigvals = der.eigvals_hessian_matrix(img, i, scale, grad_type)
        l0, l1, l2 = der.sortbyabs(eigvals, 0)
        l2m = l2.copy()
        l2m[l2 > (np.min(l2) * tau)] = np.min(l2) * tau
        blobness = np.square(l0) * l2m * (27 / (2 * l0 + l2m) ** 3)
        blobness[np.abs(l0) > np.abs(l2m)] = 1
        blobness[l0 > 0] = 0; blobness[l1 > 0] = 0; blobness[l2 > 0] = 0  
        blobness = np.nan_to_num(blobness)
        cube.append(blobness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)    
    return res * (1 - cmask) 

    

def create_hollow_selems(sizes):
    sizes = np.asarray(sizes)
    sizes = np.sort(sizes)
    boxes = []
    for size in sizes:
        ball = cnv.eselem(size, size, size) > 0
        boxes.append(ball)
    padded = []
    lengths = 2 * sizes + 1
    padders = np.diff(lengths)
    padders //= 2
    for i, (padder, box) in enumerate(zip(padders, boxes[1:])):
        pbox = boxes[i]
        pbox = np.pad(pbox, padder)
        padded.append(pbox)
    padded_full = padded.copy()
    final = [boxes[0]] 
    for box, pdd in zip(boxes[1:], padded_full):
        box[pdd] = 0
        final.append(box)
    return final   

def maxdiff_blobness(im, radii = (1, 2, 3, 4), pw = None, sig = None):
    selems = create_hollow_selems(radii)
    imcube = np.stack([ndi.maximum_filter(im, footprint = item) for item in selems], axis = 0)
    if sig is None:
        maxi = imcube[0].copy()
    else:
        maxi = ndi.gaussian_filter(im, sig)
    # maxi = cnv.rescale(maxi)
    if pw is None:
        maxdiff = np.max(maxi - imcube[1:], axis = 0)
    else:
        normaliser = np.power(maxi, pw)
        maxdiff = np.max(cnv.div_nonzero(maxi- imcube[1:], normaliser), axis = 0)
    maxdiff[maxdiff < 0] = 0
    maxdiff[im < im.mean()] = 0
    return maxdiff

###########################################################################################################################



################################################### Vesselness filters ####################################################
###########################################################################################################################

def sato_vesselness(img, sig = 1, alpha1 = 0.5, alpha2 = 2,
                    min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Sato's curvilinearity enhancement based on the paper:
        "" Sato Y, Nakajima S, Shiraga N, Atsumi H, Yoshida S, Koller T, Gerig G, Kikinis R. 
            'Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear 
            structures in medical images.' Med Image Anal. 1998 Jun;2(2):143-68. doi: 10.1016/s1361-8415(98)80009-1. PMID: 10646760. ""
            
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the thickness of the lines to be detected. 
            By passing an iterable of sigmas, lines with a variety of sizes can be detected.
        alpha1: float
            Parameter to adjust the filter's sensitivity to vessel-like objects. Should be lower than 'alpha2'.
        alpha2: float
            Parameter to adjust the filter's sensitivity to vessel-like objects. Should be higher than 'alpha1'.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        vesselness: array 
            Numpy array representing the output image with enhanced lines. Has the same shape as 'img'.
        
    """
    cnv.cp_3d(img)
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig:    
        vesselness = np.zeros_like(img)
        l0, l1, l2 = der.eigvals_hessian_matrix(img, i, scale, grad_type)
        term0 = -l1 * np.exp(-(np.square(l2) / (2 * np.square(alpha1 * l1))))  ### both l1 and l2 are smaller than 0
        term1 = -l1 * np.exp(-(np.square(l2) / (2 * np.square(alpha2 * l1)))) ### l2 is greater than 0 and l1 is smaller than 0
        vesselness[(l2 < 0) & (l1 < 0)] = term0[(l2 < 0) & (l1 < 0)] 
        vesselness[(l2 > 0) & (l1 < 0)] = term1[(l2 > 0) & (l1 < 0)] 
        vesselness[l1 >= 0] = 0
        cube.append(vesselness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)
    return res * (1 - cmask)  

def frangi_vesselness(img, sig = 1, alpha = 0.25, beta = 0.5, c = 0.1, 
                      min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Frangi's curvilinearity enhancement based on the paper:
        "" Frangi, A. F., Niessen, W. J., Vincken, K. L. & Viergever, M. A. 'Multiscale vessel enhancement filtering.'
        in Medical Image Computing and Computer-Assisted Intervention — MICCAI’98 (eds. Wells, W. M., Colchester, A. & Delp, S.) 
        vol. 1496 130–137 (Springer Berlin Heidelberg, 1998). ""
            
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the thickness of the lines to be detected. 
            By passing an iterable of sigmas, lines with a variety of sizes can be detected.
        alpha: float
            Parameter to adjust the deviation from plate-like objects.
        beta: float
            Parameter to adjust the deviation from blob-like objects. 
        c: float
            Structureness parameter. Adjusts the deviation from areas of low signal-to-noise.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        vesselness: array 
            Numpy array representing the output image with enhanced lines. Has the same shape as 'img'.
        
    """
    cnv.cp_3d(img)
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig:    
        eigvals = der.eigvals_hessian_matrix(img, i, scale, grad_type)
        eigvals_sorted = der.sortbyabs(eigvals, 0)
        l0, l1, l2 = [st._block_zeroes(item) for item in eigvals_sorted]
        rb = np.abs(l0) / np.sqrt(np.abs(l1 * l2))
        ra = np.abs(l1) / np.abs(l2)
        s = np.linalg.norm((l0, l1, l2), axis = 0)
        term0 = (1 - np.exp(-(ra ** 2) / (2 * alpha ** 2)))
        term1 = (np.exp(-(rb ** 2) / (2 * beta ** 2)))
        term2 = (1 - np.exp(-(s ** 2) / (2 * c ** 2)))
        vesselness = term0 * term1 * term2
        lmask = (eigvals_sorted[1] < 0) & (eigvals_sorted[2] < 0)
        cube.append(vesselness * lmask)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)
    return res * (1 - cmask)  


def meijering_vesselness(img, sig = 1, alpha = 0.3, min_contrast = 0.01, grad_type = 'numpy'):
    """ A simplified form of the Meijering's neuriteness enhancement method based on:
        "" Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H., Unser, M. (2004). 
        Design and validation of a tool for neurite tracing and analysis in fluorescence microscopy images. 
        Cytometry Part A, 58(2), 167-176. DOI:10.1002/cyto.a.20022 ""
                    
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the thickness of the lines to be detected. 
            By passing an iterable of sigmas, lines with a variety of sizes can be detected.
        alpha: float
            Parameter to adjust the deviation from plate-like objects.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        vesselness: array 
            Numpy array representing the output image with enhanced lines. Has the same shape as 'img'.
        
    """
    arrays = []
    cnv.cp_3d(img)
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig: 
        eigvals = der.eigvals_hessian_matrix(img, i, False, grad_type)
        l0, l1, l2 = der.sortbyabs(eigvals)
        lnorm = l2 + alpha * l0 + alpha * l1
        arrays.append(lnorm)
        summe = np.sum(arrays, 0)
        vesselness = np.where(summe < 0, summe / np.min(summe), 0)
        cube.append(vesselness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)
    return res * (1 - cmask)


def jerman_vesselness(img, sig = 1, tau = 0.5, min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Jerman's vesselness enhancement filter based on 
        "" T. Jerman, F. Pernus, B. Likar, Z. Spiclin, "Enhancement of Vascular Structures in 3D and 2D Angiographic Images", 
        IEEE Transactions on Medical Imaging, 35(9), p. 2107-2118 (2016), doi={10.1109/TMI.2016.2550102} ""

        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the thickness of the lines to be detected. 
            By passing an iterable of sigmas, lines with a variety of sizes can be detected.
        tau: float
            Parameter to adjust the sensitivity to vessel-like objects.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        vesselness: array 
            Numpy array representing the output image with enhanced lines. Has the same shape as 'img'.        
        
    """
    cnv.cp_3d(img)
    assert img.max() == 1., 'Input image is not normalised. First normalise it between 0 and 1.'
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig: 
        eigvals = - der.eigvals_hessian_matrix(img, i, scale, grad_type)
        l0, l1, l2 = der.sortbyabs(eigvals)
        lr = l2.copy()
        lr[(l2 > 0) & (l2 <= tau * np.max(l2))] = tau * np.max(l2)
        lr[l2 <= 0] = 0
        vesselness = np.square(l1) * (lr - l1) * (27 / ((l1 + lr) ** 3))
        vesselness[(l1 > (lr / 2)) & (lr > 0)] = 1
        vesselness[(l1 < 0) | (lr <= 0)] = 0
        vesselness = np.nan_to_num(vesselness)
        cube.append(vesselness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)
    return res * (1 - cmask)


def zhang_vesselness(img, sig = 1, tau = 0.5, min_contrast = 0.01, grad_type = 'numpy', scale = False):
    """ Vesselness enhancement as implemented in the paper:
        "" R. Zhang, Z. Zhou, W. Wu, C.-C. Lin, P.-H. Tsui, and S. Wu, “An improved fuzzy connectedness 
        method for automatic three-dimensional liver vessel segmentation in CT images,” 
        J Healthc Eng, vol. 2018, pp.1–18, 2018. ""
        
        Parameters:
        -----------
        img: array
            3D numpy array representing the input image
        sig: scalar or iterable
            Sigma for the hessian. Sigma here represents the thickness of the lines to be detected. 
            By passing an iterable of sigmas, lines with a variety of sizes can be detected.
        tau: float
            Parameter to adjust the sensitivity to vessel-like objects.
        min_contrast: float
            Local contrast threshold to get a contrast mask on the input image. The low-contrast areas
            are set to zero in the enhanced image that is returned.
        grad_type: str
            The method for calculating the gradients.
        scale: bool
            If True, the hessian matrix is scaled by the size of the sigma.
        
        Returns:
        --------
        vesselness: array 
            Numpy array representing the output image with enhanced lines. Has the same shape as 'img'. 
        
    """
    cnv.cp_3d(img)
    assert img.max() == 1., 'Input image is not normalised. First normalise it between 0 and 1.'
    if np.isscalar(sig):
        sig = [sig]
    cube = []
    for i in sig: 
        eigvals = - der.eigvals_hessian_matrix(img, i, scale, grad_type)
        l0, l1, l2 = der.sortbyabs(eigvals)
        lr = l2.copy()
        lr[(l2 > 0) & (l2 <= tau * np.max(l2))] = tau * np.max(l2)
        lr[l2 <= 0] = 0
        zhang_term = 1 - np.exp(-1.5 * ((np.square(l0) + np.square(l1) + np.square(lr)) / lr))
        vesselness = np.square(l1) * (lr - l1) * (27 / ((l1 + lr) ** 3)) * zhang_term 
        vesselness[(l1 > (lr / 2)) & (lr > 0)] = 1
        vesselness[(l1 < 0) | (lr <= 0)] = 0
        vesselness = np.nan_to_num(vesselness)
        cube.append(vesselness)
    res = np.max(cube, axis = 0)
    cmask = lt._generic_contrast_mask(img, 5, min_contrast)
    return res * (1 - cmask)

###########################################################################################################################



