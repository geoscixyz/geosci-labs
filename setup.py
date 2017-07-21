#!/usr/bin/env python
from __future__ import print_function
"""gpgLabs

gpgLabs are the codes that support the interactive labs provided through
http://gpg.geosci.xyz.
"""

from distutils.core import setup
from setuptools import find_packages


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open('README.rst') as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name = 'gpgLabs',
    version = '0.0.3',
    packages = find_packages(),
    install_requires = [
        'future',
        'numpy>=1.7',
        'scipy>=0.13',
        'matplotlib',
        'pandas',
        'ipywidgets',
        'properties[math]',
        'SimPEG>=0.4.1',
        'jupyter',
        'jupyter_dashboards',
        'Pillow',
        'discretize',
        'em_examples'
    ],
    author = 'GeoSci Developers',
    author_email = 'lheagy@eos.ubc.ca',
    description = 'gpgLabs',
    long_description = LONG_DESCRIPTION,
    keywords = 'geophysics, electromagnetics',
    url = 'http://gpg.geosci.xyz',
    download_url = 'https://github.com/geoscixyz/gpgLabs',
    classifiers=CLASSIFIERS,
    platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License',
    use_2to3 = False,
)
