# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:49:45 2019

@author: ozdemir
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

# local imports
from . import statistical as st
from ....utils import convenience as cnv


#################################################################    
#################################################################    
def gaussian(array, Sigma):
    return ndi.gaussian_filter(array, Sigma, mode = 'constant', cval = 0)

def sortbyabs(array, axis = 0): ### copied from skimage
    """ Sorts the values of a multidimensional array along given axis. """
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))
    index[axis] = np.abs(array).argsort(axis)
    return array[tuple(index)]

def _generic_3d_kernel(s = (1, 2, 1), d = (-1, 0, 1), axis = 2): ### where a0 = a1 so actually there are two vectors to begin with
    """ Creates a 3D gradient kernel with given 2 vectors.
        
        Parameters:
        -----------
        s: iterable
            The first vector to calculate the kernel. Typically a tuple of size 3.
        d: iterable
            The second vector to calculate the kernel. Typically a tuple of size 3.
        axis: int
            The axis, across which, the gradient is calculated. Order is in 'ij' format

        Returns:
        --------        
        kernel: array of int
            A 3D array to be used as kernel for gradient calculation. 
        """
    if axis == 2:
        h0 = np.asarray(s).reshape(1, 3) 
        h1 = np.asarray(s).reshape(1, 3) 
        h2 = np.asarray(d).reshape(1, 3) 
        kernel = h2.reshape(1, 3) * h1.reshape(3, 1) * h0.reshape(3, 1, 1)
    elif axis == 1:
        h0 = np.asarray(s).reshape(1, 3) 
        h1 = np.asarray(d).reshape(1, 3) 
        h2 = np.asarray(s).reshape(1, 3) 
        kernel = h1.reshape(3, 1) * h0.reshape(3, 1, 1) * h2.reshape(1, 3)
    elif axis == 0:
        h0 = np.asarray(d).reshape(1, 3) 
        h1 = np.asarray(s).reshape(1, 3) 
        h2 = np.asarray(s).reshape(1, 3) 
        kernel = h0.reshape(3, 1, 1) * h2.reshape(1, 3) * h1.reshape(3, 1)
    return kernel


kernels = {'sobel': (_generic_3d_kernel(s = (1,2,1), d = (-1,0,1), axis = 0),
                     _generic_3d_kernel(s = (1,2,1), d = (-1,0,1), axis = 1),
                     _generic_3d_kernel(s = (1,2,1), d = (-1,0,1), axis = 2)),
           'prewitt': (_generic_3d_kernel(s = (1,1,1), d = (-1,0,1), axis = 0),
                       _generic_3d_kernel(s = (1,1,1), d = (-1,0,1), axis = 1),
                       _generic_3d_kernel(s = (1,1,1), d = (-1,0,1), axis = 2)),
           'scharrv1': (_generic_3d_kernel(s = (3,10,3), d = (-3,0,3), axis = 0),
                        _generic_3d_kernel(s = (3,10,3), d = (-3,0,3), axis = 1),
                        _generic_3d_kernel(s = (3,10,3), d = (-3,0,3), axis = 2)),
           'scharrv2': (_generic_3d_kernel(s = (47,162,47), d = (-47,0,47), axis = 0),
                        _generic_3d_kernel(s = (47,162,47), d = (-47,0,47), axis = 1),
                        _generic_3d_kernel(s = (47,162,47), d = (-47,0,47), axis = 2))
            }

    

def generic_3d_gradient(img, s = (1,2,1), d = (-1, 0, 1), return_magnitude = False, sig = None):
    """ Calculates the gradient of the input image using a custom kernel defined with the given vectors.
        
        Parameters:
        -----------
        s: iterable
            The first vector to calculate gradient. Typically a tuple of size 3.
        d: iterable
            The second vector to calculate gradient. Typically a tuple of size 3.
        return_magnitude: bool
            Specifies if the gradient magnitude should be returned.
        sig: scalar
            Specifies the Sigma value for the Gaussian filter. 

        Returns:
        --------        
        grad: tuple of arrays or array
            If return_magnitude is True, it is a gradient magnitude image with the same shape as input image.
            If return_magnitude is False, it is a tuple of 3 arrays, each representing the gradient over a different axis, with
            the axis order being (0, 1, 2).
        """
    if sig is not None:
        img = gaussian(img, sig)
    kern0 = _generic_3d_kernel(s, d, axis=0)
    kern1 = _generic_3d_kernel(s, d, axis=1)
    kern2 = _generic_3d_kernel(s, d, axis=2)
    grad0 = - fftconvolve(img, kern0, mode = 'same')
    grad1 = - fftconvolve(img, kern1, mode = 'same')
    grad2 = - fftconvolve(img, kern2, mode = 'same')
    if return_magnitude:
        grad = np.sqrt(np.square(grad0) + np.square(grad1) + np.square(grad2))
    else:
        grad = (grad0, grad1, grad2)
    return grad


def _3d_gradient(img, kernel_name = 'sobel', return_magnitude = False, sig = None):
    """ Calculates the gradient of the input image using a kernel defined with the kernel name.
        
        Parameters:
        -----------
        s: iterable
            The first vector to calculate gradient. Typically a tuple of size 3.
        d: iterable
            The second vector to calculate gradient. Typically a tuple of size 3.
        return_magnitude: bool
            Specifies if the gradient magnitude should be returned.
        sig: scalar
            Specifies the Sigma value for the Gaussian filter. Needed only if 'return_magnitude' is True

        Returns:
        --------        
        grad: Tuple of arrays or array
            If return_magnitude is True, it is a gradient magnitude image with the same shape as input image.
            If return_magnitude is False, it is a tuple of 3 arrays, each representing the gradient over a different axis, with
            the axis order being (0, 1, 2).
        """
    cnv.cp_3d(img)
    if sig is not None:
        img = gaussian(img, sig)
    kern0, kern1, kern2 = kernels[kernel_name]
    grad0 = - fftconvolve(img, kern0, mode = 'same')
    grad1 = - fftconvolve(img, kern1, mode = 'same')
    grad2 = - fftconvolve(img, kern2, mode = 'same')
    if return_magnitude:
        grad = np.sqrt(np.square(grad0) + np.square(grad1) + np.square(grad2))
    else:
        grad = (grad0, grad1, grad2)
    return grad

def _get_eigvals(matrices):
    """ Calculates the eigenvalues from the input matrices and transposes them for convenience. """
    eigvals = np.linalg.eigvalsh(matrices)
    transposed = np.transpose(eigvals, (eigvals.ndim - 1,) + tuple(range(eigvals.ndim - 1)))
    return transposed


def gaussian_sobel_magnitude(img, sig = None):
    return _3d_gradient(img, 'sobel', True, sig)

def gaussian_sobel(img, sig = None):
    return _3d_gradient(img, 'sobel', False, sig)
    
def gaussian_prewitt_magnitude(img, sig = None):
    return _3d_gradient(img, 'prewitt', True, sig)

def gaussian_prewitt(img, sig = None):
    return _3d_gradient(img, 'sobel', False, sig)

def gaussian_scharrv1_magnitude(img, sig = None):
    return _3d_gradient(img, 'scharrv1', True, sig)

def gaussian_scharrv1(img, sig = None):
    return _3d_gradient(img, 'scharrv1', False, sig)

def gaussian_scharrv2_magnitude(img, sig = None):
    return _3d_gradient(img, 'scharrv2', True, sig)

def gaussian_scharrv2(img, sig = None):
    return _3d_gradient(img, 'scharrv2', False, sig)


def _get_gradient_method(gradient_type = 'numpy'):
    """ Convenience function to select a gradient method. """
    if gradient_type == 'numpy':
        gradient = np.gradient
    elif gradient_type == 'sobel':
        gradient = gaussian_sobel
    elif gradient_type == 'prewitt':
        gradient=gaussian_prewitt
    elif gradient_type=='scharrv1':
        gradient=gaussian_scharrv1
    elif gradient_type=='scharrv2':
        gradient=gaussian_scharrv2
    else:
        raise TypeError('The given gradient_type is invalid.')
    return gradient


def generate_structure_tensor(img, fp = 1, gradient_type='numpy', weight_func = gaussian, pregradient_sig = 0):  # ideally, presmoothing should be false
    """ Calculates the structure tensor (autocorrelation matrix) of an image.
        
        Parameters:
        -----------
        img: array
            A 3D numpy array.
        fp: scalar or iterable of 3
            The edge lengths for the local window (or 'Sigma' if 'weight_func' is a Gaussian filter).
        gradient_type: str
            The method for calculating the gradients.
        weight_func: function
            The weighting function, which should be a local statistical filter. Gaussian by default
        pregradient_sig: Sigma for a Gaussian filter that is to be applied before calculating the gradients.
            If 0, then gradients are calculated without applying a Gaussian smooth in advance (default case).
            
        Returns:
        --------        
        structure_tensor: array
            A 5D numpy array, where the last 2 dimensions specify the structure tensor values for each voxel.
        """    
    cnv.cp_3d(img)
    gradient = _get_gradient_method(gradient_type)
    structure_tensor = np.zeros(img.shape + (img.ndim, img.ndim))
    if pregradient_sig > 0:
        convolved = gaussian(img, Sigma = pregradient_sig)
        grads = gradient(convolved)
    else:
        grads = gradient(img)
    grads = np.array(grads)
    dims = np.arange(len(grads))
    coords = cnv.cartesian(dims, dims)
    items0 = grads[coords[:, 0]]
    items1 = grads[coords[:, 1]]    
    products = items0 * items1
    for coo, product in zip(coords, products):
        i, j = coo
        structure_tensor[..., i, j] = weight_func(product, fp)
    return structure_tensor


def eigvals_structure_tensor(img, fp = 1, scale = False, gradient_type='numpy', weight_func = gaussian, pregradient_sig = 0):
    """ Calculates the eigenvalues of the structure tensor of an image.
        
        Parameters:
        -----------
        img: array
            A 3D numpy array.
        fp: scalar or iterable of 3
            The edge lengths for the local window (or 'Sigma' if 'weight_func' is a Gaussian filter).
        scale: bool
            If True, the structure tensor is scaled by the size of the structuring element 'fp'
        gradient_type: str
            The method for calculating the gradients.
        weight_func: function
            The weighting function, which should be a local statistical filter. Gaussian by default
        pregradient_sig: Sigma for a Gaussian filter that is to be applied before calculating the gradients.
            If 0, then gradients are calculated without applying a Gaussian smooth in advance (default case).
            
        Returns:
        --------        
        eigenvalues: array
            A 4D numpy array, where the three 3D structure tensors are concatenated along the first axis. 
        """
    cnv.cp_3d(img)
    matrices = generate_structure_tensor(img, fp, gradient_type, weight_func, pregradient_sig)
    if scale:
        if np.isscalar(fp):
            l = fp
        else:
            l = fp[0]
        matrices = matrices * np.square(l) 
    return _get_eigvals(matrices)



######################################################### Hessian matrix #########################################################
##################################################################################################################################

def generate_hessian_matrix (img, pregradient_sig = 1, gradient_type = 'numpy', sig = 0): ### ideally no post-gradient smoothing (sig = 0)
    """ Calculates the hessian matrix of an image.
        
        Parameters:
        -----------
        img: array
            A 3D numpy array.
        pregradient_sig: scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied before calculating the gradients.
            If 0, then gradients are calculated without applying a Gaussian smooth in advance.
        gradient_type: str
            The method for calculating the gradients.
        sig:  scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied to the first gradients. Note that
            this is an illegitimate operation. If 0, this filtering step is skipped (default and strongly advised).
            
        Returns:
        --------        
        hessian_matrix: array
            A 5D numpy array, where the last 2 dimensions specify the hessian matrix for each voxel.
        """
    gradient = _get_gradient_method(gradient_type)
    hessian_matrix = np.zeros(img.shape + (img.ndim, img.ndim))
    if pregradient_sig > 0:
        convolved = gaussian(img, Sigma = pregradient_sig)
        grads = gradient(convolved)
    else:
        grads = gradient(img)
    for i0, item0 in enumerate(grads):
        if sig > 0:
            convolved = gaussian(item0, Sigma = sig)
        else:
            convolved = item0.copy()
        second_grad = gradient(convolved)
        for i1, item1 in enumerate(second_grad):
            hessian_matrix[..., i0, i1] = item1
    return hessian_matrix


def eigvals_hessian_matrix(img, pregradient_sig = 1, scale = False, gradient_type = 'numpy', sig = 0): 
    """ Calculates the eigenvalues of the hessian matrix of an image.
        
        Parameters:
        -----------
        img: array
            A 3D numpy array.
        pregradient_sig: scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied before calculating the gradients.
            If 0, then gradients are calculated without applying a Gaussian smooth in advance.
        scale: bool
            If True, the hessian matrix is scaled by the size of the pregradient_Sigma 
        gradient_type: str
            The method for calculating the gradients.
        sig:  scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied to the first gradients. Note that
            this is an illegitimate operation. If 0, this filtering step is skipped (default and strongly advised).
            
        Returns:
        --------        
        eigenvalues: array
            A 4D numpy array, where the three 3D eigenvalue arrays are concatenated along the first axis. 
        """
    cnv.cp_3d(img)
    matrices = generate_hessian_matrix(img, pregradient_sig, gradient_type, sig)
    if scale: 
        matrices = matrices * np.square(pregradient_sig) 
    return _get_eigvals(matrices)  


def determinant_hessian_matrix(img, pregradient_sig = 1, scale = False, gradient_type = 'numpy', sig = 0):
    """ Calculates the determinant of the hessian matrix of an image. Note that this is a popular blob detector (DoH filter).
        
        Parameters:
        -----------
        img: array
            A 3D numpy array.
        pregradient_sig: scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied before calculating the gradients.
            If 0, then gradients are calculated without applying a Gaussian smooth in advance.
        scale: bool
            If True, the hessian matrix is scaled by the size of the pregradient_Sigma 
        gradient_type: str
            The method for calculating the gradients.
        sig:  scalar or iterable of size 3
            Sigma for a Gaussian filter that is to be applied to the first gradients. Note that
            this is an illegitimate operation. If 0, this filtering step is skipped (default and strongly advised).
            
        Returns:
        --------        
        determinants: array
            A numpy array with the determinants for each voxel. 
        """
    cnv.cp_3d(img)
    matrices = generate_hessian_matrix(img, pregradient_sig, gradient_type, sig)
    if scale: 
        matrices = matrices * np.square(pregradient_sig) 
    return np.linalg.det(matrices)














