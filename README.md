# ZarrSeg
Command line tool to segment local and remote OME-Zarr data

### Example command line to threshold local OME-Zarr:

```bash
zseg threshold -m otsu -c 1 -ch 0 -n otsu-c1-ch0 --colormap viridis /path/to/example.zarr
```

The command line arguments: 

`-m`: the threshold method, which here is the Otsu method.\
`-c`: the coefficient, that is multiplied with the threshold found by the method specified with `-m`.\
`-ch`: the channel that is to be thresholded. Needed if the input OME-Zarr has multiple channels.\
`-n`: the name of the output. This will be the folder name created for this segmentation under the `labels` path\
`--colormap`: colormap that is used to display the label image with `napari-ome-zarr`. Any of the matplotlib colormaps can be specified.
