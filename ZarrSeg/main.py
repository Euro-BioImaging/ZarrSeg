#!/usr/bin/env python3

import argparse, os, sys, json

scriptpath = os.path.dirname(os.path.realpath(__file__))
nestedpath0 = os.path.join(scriptpath, 'bin')
nestedpath1 = os.path.join(scriptpath, 'bin/transforms')
nestedpath2 = os.path.join(scriptpath, 'bin/transforms', 'photometric')
sys.path.extend([scriptpath, nestedpath0, nestedpath1, nestedpath2])

from ZarrSegment import ZarrSegment

if __name__ == '__main__':
    s3_params = ["endpoint", "key", "secret", "region"]
    commands = ['threshold', 'postprocess', 'configure_s3_remote']

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    configure_s3_remote = subparsers.add_parser('configure_s3_remote')
    configure_s3_remote.add_argument('--url', default = None)
    configure_s3_remote.add_argument('--access', default = None)
    configure_s3_remote.add_argument('--secret', default = None)
    configure_s3_remote.add_argument('--region', default = None)

    threshold_parser = subparsers.add_parser('threshold')
    threshold_parser.add_argument('--method', '-m', default = 'otsu', help = 'Thresholding method', choices = ['isodata', 'li', 'manual_threshold', 'otsu', 'ridler', 'ridler_median', 'ridler_wmean', 'yen'])
    threshold_parser.add_argument('--coef', '-c', default = 1, type = float, help = 'Coefficient that is multiplied with the calculated value to get the effective threshold.')
    threshold_parser.add_argument('--name', '-n', default = '0', help = "Name of the OME-Zarr subdirectory corresponding to the labels calculated with this command.")
    threshold_parser.add_argument('--channel', '-ch', default = None, type = int, help = "Image channel to be used for segmentation. Needs to be specified if there are multiple channels.")
    threshold_parser.add_argument('--colormap', '-cm', default = 'viridis', help = "RGB values to be saved to the labels metadata file. This colormap is used by default by the OME-Zarr viewers.")
    threshold_parser.add_argument('--remote', '-r', default = False, action='store_true', help = "If True, fpath is assumed to be in the configured s3 endpoint.")
    threshold_parser.add_argument('fpath', help = "OME-Zarr path to use")
    postprocess_parser = subparsers.add_parser('postprocess')
    postprocess_parser.add_argument('--method', '-m', default = 'binary_opening', help = 'Mathematical morphology method', choices = ['binary_closing', 'binary_dilation', 'binary_erosion', 'binary_opening'])
    postprocess_parser.add_argument('--footprint', '-f', default = '1,1', help = "Shape of the footprint for the mathematical morphology operation. Must be comma-separated list of integers of length 2 or 3")
    postprocess_parser.add_argument('--min_size', '-ms', default = 64, type = int, help = "Minimum size of the labels to be accepted. Labels with voxel numbers less than this will be deleted.")
    postprocess_parser.add_argument('--labels_tag', '-l', default = '0', help = "Name of the labels, on which postprocessing is performed.")
    postprocess_parser.add_argument('--colormap', '-cm', default = 'viridis', help = "RGB values to be saved to the labels metadata file. This colormap is used by default by the OME-Zarr viewers.")
    postprocess_parser.add_argument('--remote', '-r', default = False, action='store_true', help = "If True, fpath is assumed to be in the configured s3 endpoint.")
    postprocess_parser.add_argument('fpath', help = "OME-Zarr path to use")

    args = parser.parse_args()

    if len(sys.argv) > 1:
        prompt = str(sys.argv[1])
        if prompt not in commands:
            raise ValueError('Command must be either of: {}'.format(commands))
    else:
        raise ValueError('Command is missing. Either of the following commands is needed.{}'.format(commands))

    if prompt == 'configure_s3_remote':
        url_prompt = 'enter url:\nEnter "skip" or "s" if you would like to keep the current value\n'
        access_prompt = 'enter access key:\nEnter "skip" or "s" if you would like to keep the current value\n'
        secret_prompt = 'enter secret key:\nEnter "skip" or "s" if you would like to keep the current value\n'
        region_prompt = 'enter region name\nEnter "skip" or "s" if you would like to keep the current value\n'

        if args.url is None:
            args.url = input(url_prompt)
        if args.access is None:
            args.access = input(access_prompt)
        if args.secret is None:
            args.secret = input(secret_prompt)
        if args.region is None:
            args.region = input(region_prompt)
        # print(args)
        jsonpath = os.path.join(scriptpath,  'configs', 's3credentials')
        if not os.path.exists(jsonpath):
            with open(jsonpath, 'a+') as f:
                primer = {'endpoint': 'placehold'}
                json.dump(primer, f, indent = 2)

        with open(jsonpath, 'r+') as f:
            jsonfile = json.load(f)
            # jsondict = dict(jsonfile)
            for i, (_, value) in enumerate(args.__dict__.items()):
                key = s3_params[i]
                if (value == 's') | (value == 'skip'):
                    pass
                elif len(value) == 0:
                    try:
                        del jsonfile[key]
                    except:
                        pass
                elif len(value) > 0:
                    if key == 'endpoint' and not value.startswith('https://'):
                        value = 'https://' + value
                    jsonfile[key] = value
            # print(jsonfile)
            f.seek(0)
            json.dump(jsonfile, f, indent = 2)
            f.truncate()
        print("Configuration of the default s3 credentials for 'zseg' is complete.")

    ### Acquire the credentials and context already here
    configpath = os.path.join(scriptpath,  'configs', 's3credentials')
    if os.path.exists(configpath):
        with open(configpath, 'r+') as f:
            credentials = json.load(f)
    else:
        credentials = None

    context = 'local'
    if prompt == 'threshold' or prompt == 'postprocess':
        if args.remote:
            context = 'remote'
            if credentials is None:
                raise ValueError('Remote segmentation is specified but no remote configuration file is detected.')
    ### Credentials and context acquired

    if prompt == 'threshold':
        zseg = ZarrSegment(args.fpath, context = context, credentials = credentials)
        zseg.threshold(method = args.method,
                       coef = args.coef,
                       labels_name = args.name,
                       channel = args.channel,
                       colormap = args.colormap,
                       )

    elif prompt == 'postprocess':
        zseg = ZarrSegment(args.fpath, context = context, credentials = credentials)
        fp = args.footprint.split(',')
        fp = [int(i) for i in fp]
        zseg.postprocess(labels_tag = args.labels_tag,
                         method = args.method,
                         footprint = fp,
                         min_size = args.min_size,
                         colormap = args.colormap,
                         )


