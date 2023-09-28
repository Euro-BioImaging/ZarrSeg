# -*- coding: utf-8 -*-
"""
@author: bugra
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


def parse_requirements(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]
    return requires

def readme():
   with open('README.txt') as f:
       return f.read()

requirements = parse_requirements('requirements.txt')

setuptools.setup(
    name = 'ZarrSeg',
    version = '0.0.1',
    author = 'Bugra Özdemir',
    author_email = 'bugraa.ozdemir@gmail.com',
    description = 'A package for segmentation of OME-Zarr datasets.',
    long_description = readme(),
    long_description_content_type = "text/markdown",
    include_package_data = True,
    url = 'https://github.com/Euro-BioImaging/ZarrSeg',
    # license = 'MIT',
    packages = setuptools.find_packages(),
    install_requires = requirements
    )
