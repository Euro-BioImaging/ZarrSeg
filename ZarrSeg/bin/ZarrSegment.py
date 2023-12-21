from pathlib import Path
import numpy as np, inspect, os
from skimage import filters, morphology
from skimage.draw import ellipsoid

from DataLoader import ZarrSampler, index_nth_dimension
from transforms.photometric.thresholding import global_threshold as gt

scriptpath = os.path.dirname(os.path.realpath(__file__))

class ZarrSegment(ZarrSampler):
    threshold_methods = [item for item in dir(gt) if not item.startswith('_') and item != 'np']
    def __init__(self,
                root_path: [str, Path],
                backup: bool = False,
                credentials: [dict, None, str, Path] = None, # TODO: Add an option to parse this from a json file stored in a local path.
                context: str = 'default',
                subset_dim = None,
                subset_idx = None
                ):

        ZarrSampler.__init__(self, root_path, backup, credentials, context)
        self.__resample()
        self.__subset_dim = subset_dim
        self.__subset_idx = subset_idx
        self.__reset_argdict()

    def __resample(self):
        # self.sample = self.multiscls[0]
        self.sample = np.array(self.array)

    def __set_subset_axis(self):
        if self.subset_dim is None:
            self.__ax = None
        elif self.subset_idx is None:
            self.__ax = None
        else:
            axes = self.dimensions
            self.__ax = axes.index(self.subset_dim)  ### Dimension of selecting the data for segmentation
        if 'c' in self.dimensions and self.c_size > 1:
            if self.__ax is None:
                raise ValueError("Note that there are multiple channels and but no specific channel has been selected (i.e., trying to segment the entire series.)\n"
                                "This is currently not supported. Consider selecting a specific channel by specifying the 'channel' parameter.")

    @property
    def subset_dim(self):
        assert type(self.__subset_dim) in [str, type(None)], 'Subset dimension must be of type str'
        return self.__subset_dim
    @subset_dim.setter
    def subset_dim(self,
                   subset_dim: [str, None]
                   ):
        assert type(subset_dim) in [str, type(None)], 'Subset dimension must be of type str'
        self.__subset_dim = subset_dim
        self.__set_subset_axis()
        print(self.__ax)

    @property
    def subset_idx(self):
        assert type(self.__subset_idx) in [int, type(None)], 'Subset dimension must be of type int'
        return self.__subset_idx

    @subset_idx.setter
    def subset_idx(self,
                   subset_idx: [int, None]
                   ):
        assert type(subset_idx) in [int, type(None)], 'Subset dimension must be of type int'
        self.__subset_idx = subset_idx
        self.__set_subset_axis()
        print(self.__ax)

    def subset_data(self):
        sdim = self.subset_dim
        sidx = self.subset_idx
        if None in (sdim, sidx):
            raise TypeError('To subset the data for segmentation, the attributes subset_dim and subset_idx must be assigned values of type str and type int, respectively.\nCurrently, at least one of the attributes is of type None')
        self.sample = index_nth_dimension(self.sample, self.__ax, sidx)

    def __reset_argdict(self):
        self.argdict = {'low': 0,
                       'high': 256,
                       'bincount': 256,
                       'tol': 0.01,
                       'return_thresholded': False,
                       'random_start': False,
                       'verbous': False,
                       }

    def set_threshold_method(self,
                            method: str = 'otsu'
                            ):
        if method not in self.threshold_methods:
            raise ValueError('The method specified must be one of the following methods: %s' % self.threshold_methods)
        self.method = getattr(gt, method)
        argspec = inspect.getfullargspec(self.method)
        self.threshold_arguments = {}
        for key, val in self.argdict.items():
            if key in argspec.args:
                self.threshold_arguments[key] = self.argdict[key]

    def __threshold(self,
                    **kwargs
                    ):
        res0 = self.sample
        thresh = self.method(res0, **self.threshold_arguments)
        mask = (res0 > (thresh * kwargs['coef']))
        lbldf = morphology.label(mask)
        lbldf = lbldf.astype(np.uint8)
        return lbldf

    def threshold(self,
                  method = 'otsu',
                  coef=1.,
                  colormap = 'viridis',
                  labels_name = '0',
                  overwrite = False,
                  channel = None,
                  **kwargs
                  ):
        """ A global segmentation of the highest resolution level from the OME-Zarr arrays. """
        if channel is not None:
            self.__set_to_channel(channel)
        sdim = self.subset_dim
        sidx = self.subset_idx
        self.__set_subset_axis()
        self.set_threshold_method(method)
        if None in (sdim, sidx):
            lbldf = self.__threshold(coef = coef, **kwargs)
        else:
            self.subset_data()
            lbldf = self.__threshold(coef = coef, **kwargs)
            lbldf = np.expand_dims(lbldf, self.__ax)

        self.add_labels(lbldf,
                        colormap,
                        overwrite = overwrite,
                        new_labels = True,
                        labels_name = labels_name
                        )
        self.__resample() ### reset the self.sample to original dimensions.

    def __set_to_channel(self, ch = 0):
        self.__subset_dim = 'c'
        self.__subset_idx = ch

    def __eselem(self, a, b, c=None, d=None):
        """ Creates an ellipsoidal structuring element in 2D, 3D and, in a specialised manner, in 4D. """
        if c is None:
            selem = ellipsoid(1, a, b)[2:-2, 1:-1, 1:-1][0]
        elif d is None:
            selem = ellipsoid(a, b, c)[1:-1, 1:-1, 1:-1]
        else:
            selem = ellipsoid(b, c, d)[1:-1, 1:-1, 1:-1]
            selem = np.array(a * [selem])
        return selem

    def postprocess(self,
                    labels_tag,
                    method = 'binary_opening',
                    footprint = [1, 1],
                    min_size = 32,
                    colormap = 'viridis'
                    ):
        methods = ['binary_closing',
                   'binary_dilation',
                   'binary_erosion',
                   'binary_opening'
                   ]
        assert method in methods, 'The specified method is not supported'
        if len(footprint) not in [2, 3]:
            raise ValueError('The footprint can either be 2- or 3-dimensional.')
        mask = self.binary_mask(labels_tag)
        fp = self.__eselem(*footprint)
        for i in range(mask.ndim):
            if fp.ndim == mask.ndim: break
            fp = np.expand_dims(fp, 0)
        process = getattr(morphology, method)
        processed = process(mask, fp)
        lbldf = morphology.label(processed)
        lbldf = morphology.remove_small_objects(lbldf, min_size)
        lbldf = morphology.label(lbldf > 0).astype(np.uint8)
        labels_name = method + '_fp{}'.format(tuple(footprint)) + '_' + labels_tag
        self.add_labels(lbldf,
                        colormap = colormap,
                        labels_name = labels_name,
                        new_labels = True
                        )
        self.__resample() ### reset the self.sample to original dimensions.


