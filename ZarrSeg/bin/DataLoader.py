import numpy as np, dask as da
import zarr, json, shutil, os, copy
import s3fs
from pathlib import Path
import ome_zarr.scale as scaler
import matplotlib as mpl
from matplotlib import pyplot as plt

# scriptpath = os.path.dirname(os.path.realpath(__file__))

def index_nth_dimension(array,
                        dimensions = 2, # a scalar or iterable
                        intervals = None # a scalar, an iterable of scalars, a list of tuple or None
                        ):
    allinds = np.arange(array.ndim).astype(int)
    if np.isscalar(dimensions):
        dimensions = [dimensions]
    if intervals is None or np.isscalar(intervals):
        intervals = np.repeat(intervals, len(dimensions))
    assert len(intervals) == len(dimensions)
    interval_dict = {item: interval for item, interval in zip(dimensions, intervals)}
    shape = array.shape
    slcs = []
    for idx, dimlen in zip(allinds, shape):
        if idx not in dimensions:
            slc = slice(dimlen)
        else:
            try:
                slc = slice(interval_dict[idx][0], interval_dict[idx][1])
            except:
                slc = interval_dict[idx]
        slcs.append(slc)
    slcs = tuple(slcs)
    indexed = array[slcs]
    return indexed

class ZarrSampler:
    def __init__(self,
                 root_path = '/home/oezdemir/PycharmProjects/elastix/data/sample.ome.zarr',
                 # series_no = 0, # Very often we handle single series, so default series name is 0
                 backup = False,
                 credentials:[dict, str, Path, None] = None,
                 context = 'default'
                 ):
        self.__credentials = credentials
        if context == 'default':
            if self.__credentials is None:
                self.context = 'local'
            else:
                if 'endpoint' in self.__credentials:
                    self.context = 'remote'
                else:
                    self.context = 'local'
        elif context in ['local', 'remote']:
            self.context = context
        self.read_data(root_path, backup = backup)
        self.__set_multiscales()
        if not self.has_labels:
            self.__initialise_labels_metadata()
    @property
    def context(self):
        return self.__context
    @context.setter
    def context(self, context):
        credentials = self.credentials
        if credentials is None:
            self.__context = 'local'
        elif isinstance(credentials, dict):
            if 'endpoint' in credentials:
                self.__context = context
            else:
                raise TypeError('Credentials must be of type dict. Context assignment is dependent on valid credentials.')
    @property
    def credentials(self):
        return self.__credentials
    @credentials.setter
    def credentials(self,
                    credentials: [dict, str, Path, None]
                    ):
        assert type(credentials) in [dict, str, Path, type(None)]
        if isinstance(credentials, dict):
            assert 'endpoint' in credentials
            self.__credentials = credentials
        elif type(credentials) in [str, Path]:
            with open(credentials, 'r+') as f:
                self.__credentials = json.load(f)


    def __read_remote(self, root_path): ### here root_path is relative to the bucket
        assert root_path is not None
        credentials = self.credentials
        if 'key' not in credentials:
            self.fs = s3fs.S3FileSystem(anon=True,
                                        client_kwargs = dict(endpoint_url = credentials['endpoint'], region_name='eu-west-2')) ### need modification
        else:
            self.fs = s3fs.S3FileSystem(key = credentials['key'],
                                        secret = credentials['secret'],
                                        endpoint_url = credentials['endpoint'],
                                        client_kwargs = dict(region_name = credentials['region']))
        self.store = s3fs.S3Map(root = root_path, ### must start with the bucket name
                                s3 = self.fs,
                                check = False)
    def __read_local(self, root_path):
        self.store = zarr.DirectoryStore(root_path)
    def read_data(self, root_path, context = None, backup = False):
        # create root group and acquire root metadata
        if context is not None:
            self.context = context
        self.root_path = root_path
        if self.context == 'remote':
            print('Reading remote data')
            self.__read_remote(self.root_path)
        else:
            if self.context == 'local':
                print('Reading local data')
                self.__read_local(self.root_path)
            else:
                raise ValueError('Context must be either local or remote.')
        self.root = zarr.group(self.store)
        if backup:
            self.__copy_to_backup(root_path)

    @property
    def has_labels(self):
        if hasattr(self.multiscls, 'labels'):
            return True
        else:
            return False

    @property
    def has_labeled_arrays(self):
        res = False
        if hasattr(self.multiscls, 'labels'): # Labels directory was created
            if len(self.multiscls['labels']) > 0: # There is some subdirectory in the Labels directory. These must be individual label paths
                firsttag = list(self.multiscls['labels'])[0] # The first label path is selected
                if '0' in self.multiscls['labels'][firsttag]: # That label path contains at least one resolution level.
                    if isinstance(self.multiscls['labels'][firsttag][0], zarr.core.Array): # That resolution level is of zarr array type. Validation successfull.
                        res = True
        return res

    def __copy_to_backup(self, root_path):
        base = root_path.split('/')[:-1]
        name = root_path.split('/')[-1]
        backup_path = '/' + os.path.join(*base, '__.' + name)
        shutil.copytree(root_path, backup_path)
        self.backup_path = backup_path

    def __set_multiscales(self):
        if 'multiscales' in dict(self.root.attrs):
            self.multiscls = self.root
        else:
            if 'multiscales' in dict(self.root['0'].attrs):
                self.multiscls = self.root['0']
            else:
                raise ValueError('multiscales metadata could not be found.')
        self.dimension_separator = self.multiscls[0]._dimension_separator

    def __initialise_labels_metadata(self, force = False): ### Note that this is only to create metadata from the raw data. Nothing related to root and binary data for labels
        if self.has_labeled_arrays:
            if force:
                self.__labels_chunk_dims = self.chunk_dims
                self.__labels_multiscales_metadata = copy.deepcopy(self.multiscales_metadata)
                self.__labels_tags = []
            else:
                raise Warning('If labels exist, this method can only be used with parameter force being True.')
        else:
            self.__labels_chunk_dims = self.chunk_dims
            self.__labels_multiscales_metadata = copy.deepcopy(self.multiscales_metadata)
            self.__labels_tags = []

    @property
    def labels_tags(self): ### TODO IS THIS CORRECT?
        if self.has_labeled_arrays:
            self.__labels_tags = list(self.labels.group_keys())
        else:
            self.__initialise_labels_metadata()
        return self.__labels_tags

    @property
    def labels_chunk_dims(self):
        if self.has_labeled_arrays:
            tag = self.labels_tags[0]
            self.__labels_chunk_dims = self.multiscls['labels'][tag][0].chunks
        else:
            self.__initialise_labels_metadata()
        return self.__labels_chunk_dims

    def labels_multiscales_metadata(self, tag):
        if self.has_labels:
            if tag in self.labels_tags:
                self.__labels_multiscales_metadata = self.labels[tag].attrs['multiscales']
            else:
                self.__initialise_labels_metadata(True)
        else:
            self.__initialise_labels_metadata()
        return self.__labels_multiscales_metadata

    def __delitem__(self, idx):
        ### delete a resolution level ###
        multiscls_attrs = self.multiscls.attrs['multiscales']
        multiscls_attrs[0]['datasets'].pop(idx)
        self.multiscls.attrs['multiscales'] = multiscls_attrs
        # sampler.multiscls.attrs['multiscales'][0]['datasets'][2]
        del self.multiscls[idx]

    def __getitem__(self, idx):
        ### get a resolution level ###
        return self.multiscls[idx]

    @property
    def name(self):
        if 'name' in self.multiscls.attrs['multiscales'][0]:
            name = self.multiscls.attrs['multiscales'][0]['name']
        else:
            name = 'image'
        return name
    @name.setter
    def name(self, name):
        self.multiscls.attrs['multiscales'][0]['name'] = name

    @property
    def multiscales_metadata(self):   ### TODO: TAKE CARE OF ADDING NEW MULTISCALES DATA AND METADATA. MAKE IT CONDITIONAL, ONLY ADD IF THERE IS NOTHING. THIS ENABLES CREATING NEW MULTISCALES DATA IF NONE EXISTS.
        return self.multiscls.attrs['multiscales']

    @property
    def datasets_metadata(self): ### TODO: ONCE DE NOVO MULTISCALES DATA CAN BE ADDED, ALL THESE WILL AUTOMATICALLY BE DERIVED FROM THEM.
        return self.multiscls.attrs['multiscales'][0]['datasets']

    @property
    def num_resolutions(self):
        return len(self.datasets_metadata)

    @property
    def chunk_dims(self):
        return self.multiscls[0].chunks

    @property
    def datatype(self):
        return self.multiscls[0].dtype

    @property
    def compressor(self):
        return self.multiscls[0].compressor

    @property
    def axes(self):
        return self.multiscls.attrs['multiscales'][0]['axes']

    @property
    def scales(self):
        ### Physical measure of the scale ###
        scales = [dataset['coordinateTransformations'][0]['scale'] for dataset in self.datasets_metadata]
        return scales

    @property
    def units(self):
        units = []
        for ax in self.axes:
            if 'unit' in ax:
                units.append(ax['unit'])
            else:
                units.append('na')
        return units

    @property
    def dimensions(self):
        try:
            dimensions = [ax['name'] for ax in self.axes]
            dimensions = ''.join(dimensions)
        except:
            dimrange = 'tczyx'
            ndim = self.multiscls[0].ndim
            dimensions = dimrange[-ndim:]
        return dimensions

    @property
    def shape(self):
        return self.array.shape

    @property
    def t_idx(self):
        if 't' in self.dimensions:
            return self.dimensions.index('t')
        else:
            raise ValueError('The dimension t does not exist in this OME-Zarr')
    @property
    def c_idx(self):
        if 'c' in self.dimensions:
            return self.dimensions.index('c')
        else:
            raise ValueError('The dimension c does not exist in this OME-Zarr')

    @property
    def c_size(self):
        if 'c' in self.dimensions:
            return self.shape[self.c_idx]
        else:
            raise ValueError('The dimension c does not exist in this OME-Zarr')

    @property
    def z_idx(self):
        if 'z' in self.dimensions:
            return self.dimensions.index('z')
        else:
            raise ValueError('The dimension z does not exist in this OME-Zarr')
    @property
    def y_idx(self):
        if 'y' in self.dimensions:
            return self.dimensions.index('y')
        else:
            raise ValueError('The dimension y does not exist in this OME-Zarr')
    @property
    def x_idx(self):
        if 'x' in self.dimensions:
            return self.dimensions.index('x')
        else:
            raise ValueError('The dimension x does not exist in this OME-Zarr')

    @property
    def t_scale(self):
        ### t_scale for the full resolution level
        return self.scales[0][self.t_idx]
    @t_scale.setter
    def t_scale(self, newscale): ### Note that you are updating the full resolution scale, the other resolutions are downscaled automatically
        scalerange = 2 ** np.arange(len(self.scales))
        multimeta = self.multiscales_metadata
        for i in range(len(self.scales)):
            multimeta[0]['datasets'][i]['coordinateTransformations'][0]['scale'][self.t_idx] = newscale * scalerange[i]
        self.multiscls.attrs['multiscales'] = multimeta

    @property
    def c_scale(self):
        ### c_scale for the full resolution level
        return self.scales[0][self.c_idx]
    @c_scale.setter
    def c_scale(self, newscale): ### Note that you are updating the full resolution scale, the other resolutions are downscaled automatically
        scalerange = 2 ** np.arange(len(self.scales))
        multimeta = self.multiscales_metadata
        for i in range(len(self.scales)):
            multimeta[0]['datasets'][i]['coordinateTransformations'][0]['scale'][self.c_idx] = newscale * scalerange[i]
        self.multiscls.attrs['multiscales'] = multimeta

    @property
    def z_scale(self):
        ### z_scale for the full resolution level
        return self.scales[0][self.z_idx]
    @z_scale.setter
    def z_scale(self, newscale): ### Note that you are updating the full resolution scale, the other resolutions are downscaled automatically
        scalerange = 2 ** np.arange(len(self.scales))
        multimeta = self.multiscales_metadata
        for i in range(len(self.scales)):
            multimeta[0]['datasets'][i]['coordinateTransformations'][0]['scale'][self.z_idx] = newscale * scalerange[i]
        self.multiscls.attrs['multiscales'] = multimeta

    @property
    def y_scale(self):
        ### y_scale for the full resolution level
        return self.scales[0][self.y_idx]
    @y_scale.setter
    def y_scale(self, newscale): ### Note that you are updating the full resolution scale, the other resolutions are downscaled automatically
        scalerange = 2 ** np.arange(len(self.scales))
        multimeta = self.multiscales_metadata
        for i in range(len(self.scales)):
            multimeta[0]['datasets'][i]['coordinateTransformations'][0]['scale'][self.y_idx] = newscale * scalerange[i]
        self.multiscls.attrs['multiscales'] = multimeta

    @property
    def x_scale(self):
        ### x_scale for the full resolution level
        return self.scales[0][self.x_idx]
    @x_scale.setter
    def x_scale(self, newscale): ### Note that you are updating the full resolution scale, the other resolutions are downscaled automatically
        scalerange = 2 ** np.arange(len(self.scales))
        multimeta = self.multiscales_metadata
        for i in range(len(self.scales)):
            multimeta[0]['datasets'][i]['coordinateTransformations'][0]['scale'][self.x_idx] = newscale * scalerange[i]
        self.multiscls.attrs['multiscales'] = multimeta

    @property
    def array(self):
        ### Highest resolution array from the OME-Zarr series
        copied = np.array(self.multiscls[0])
        return copied
    @array.setter
    def array(self, item):
        return TypeError('Assignment is currently not supported for this attribute.')

    @property
    def mask(self):
        if 'mask' not in self.multiscls:
            raise TypeError('No mask exists.')
        return self.multiscls['mask'][0]
    @mask.setter
    def mask(self, mask):
        assert isinstance(mask, np.ndarray)
        self.masks_resampled = scaler.Scaler(max_layer = self.num_resolutions - 1).nearest(mask)
        multimeta = copy.copy(self.multiscls.attrs['multiscales'])
        # maskmeta = [{'segmentation_method': 'thresholding', 'algorithm': 'otsu', 'source': 'skimage.filters'}]
        mask_path = os.path.join(self.multiscls.path, 'mask')
        self.mask_root = self.root.create_group(mask_path)
        self.mask_datasets = {}
        for i, mask in enumerate(self.masks_resampled):
            self.mask_datasets[i] = self.mask_root.create_dataset(name = '%s' % i,
                                                                  data = mask,
                                                                  chunks = self.chunk_dims,
                                                                  compressor = self.compressor
                                                                  )
        self.mask_root.attrs['multiscales'] = [multimeta]
        # self.mask_root.attrs['mask'] = [maskmeta]

    @property
    def mask_metadata(self):
        return self.mask_root.attrs['mask']
    @mask_metadata.setter
    def mask_metadata(self, mask_metadata):
        assert isinstance(mask_metadata, list), 'Mask metadata must be an instance of list type.'
        self.mask_root.attrs['mask'] = mask_metadata

    @property
    def labels(self):
        if 'labels' not in self.multiscls:
            raise TypeError('No labels exists.')
        return self.multiscls['labels']
    @labels.setter
    def labels(self, labels): ### TODO: THIS IS THE OPTION FOR HAVING ALL LABELS IN THE SAME ARRAY. ALSO ADD THE SPLIT-LABELS OPTION.
        raise TypeError('Labels cannot be set externally. Use add_labels method.')

    def binary_mask(self, tag):
        return np.array(self.labels[tag][0]) > 0

    # @property
    # def labels_root(self):
    #     if 'labels' not in self.multiscls:
    #         raise TypeError('No labels exists.')
    #     return self.multiscls['labels']
    # @labels_root.setter
    # def labels_root(self, labels_root):
    #     raise TypeError('Labels cannot be set externally. Use add_labels method.')

    def labels_multiscales_root(self, tag):
        if 'labels' not in self.multiscls:
            raise TypeError('No labels exists.')
        elif tag not in self.labels_tags:
            raise ValueError('The specified tag does not exist')
        return self.labels[tag]

    def __create_labels_path(self,
                             labels_name: [str, None],
                             overwrite = False,
                             new_labels = True
                             ):
        if labels_name is None:
            labels_name = 'labels_for_' + self.name
        # self.__initialise_labels_metadata()
        labels_path = os.path.join(self.multiscls.path, 'labels')
        labels_multiscls_path = os.path.join(self.multiscls.path, 'labels', labels_name)
        if self.has_labels:
            if overwrite:
                self.labels_root = self.root.create_group(labels_path, overwrite = overwrite)
                print('Root label path created, overwriting all existing label paths.')
            else:
                self.labels_root = self.labels
                print('Root label path derived from existing root label path.')
        else:
            self.labels_root = self.root.create_group(labels_path, overwrite = overwrite)
            print('Root label path created de novo.')
        if not self.has_labels or overwrite or new_labels:
            self.labels_mroot = self.root.create_group(labels_multiscls_path, overwrite = overwrite) ### Note that this refers to a newly created root.

    def add_labels(self,
                   labels,
                   colormap = 'viridis',
                   labels_name = None,
                   overwrite = False,
                   new_labels = False
                   ):
        ### NOTE THAT THERE IS NO NEED TO SET LABELS METADATA SEPARATELY. THIS METHOD AUTOMATICALLY HANDLES THE METADATA.
        assert isinstance(labels, np.ndarray)
        if overwrite:
            self.__initialise_labels_metadata()
        if labels_name is None:
            labels_name = 'labels_for_' + self.name
        self.labels_tags.append(labels_name)
        self.__create_labels_path(labels_name, overwrite, new_labels)

        display = self.__get_display(labels, colormap)
        self.labels_resampled = scaler.Scaler(max_layer = self.num_resolutions - 1).nearest(labels)

        for i, lbl in enumerate(self.labels_resampled):
            self.labels_mroot.create_dataset(name = '%s' % i,
                                             data = lbl,
                                             chunks = self.labels_chunk_dims,
                                             compressor = self.compressor,
                                             dimension_separator = self.dimension_separator
                                             )
        self.labels_root.attrs['labels'] = self.labels_tags
        multimeta = copy.deepcopy(self.multiscales_metadata)
        multimeta[0]['name'] = labels_name
        # print(multimeta)
        self.labels_mroot.attrs['multiscales'] = multimeta ###
        self.labels_mroot.attrs['image-label'] = display

    def drop_labels(self): # TODO
        pass

    @property
    def labels_metadata(self): ### !!! This cannot be set by the user. !!!
        return self.labels_root.attrs['labels']

    def tree(self):
        print(self.root.tree(True))

    def rescale(self): ### create new resolution levels from the top resolution, note that whole metadata will change
        pass

    @property
    def __allcmaps(self):
        return plt.colormaps()

    def __get_display(self, lbld, colormap = 'viridis'): # TODO: fix label t0 colour map.
        lim = 255
        uqs, sizes = np.unique(lbld, return_counts = True)
        num = len(uqs)
        assert colormap in self.__allcmaps, "colormap must be one of the items in 'cmaps'. Run 'plt.colormaps()' for the full list."
        cmap = mpl.colormaps.get_cmap(colormap)
        idx = np.linspace(0, lim, num).astype(int)
        if hasattr(cmap, 'colors'):
            colors = (np.array(cmap.colors) * lim * 1.)[idx]
        else:
            colors = np.array([cmap(i) for i in (idx / lim)]) * lim * 1.

        uqs = uqs[np.argsort(sizes)]
        uqs = np.array(([uqs[-1].tolist()] + uqs[:-1].tolist()))
        display_metadata = {
                                "colors": [
                                    {"label-value": i0, "rgba": item.tolist() }
                                    for i0, item in zip(uqs, colors)
                                ]
                            }
        self.display_metadata = display_metadata.copy()
        return display_metadata