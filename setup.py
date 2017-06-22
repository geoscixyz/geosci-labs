#!/usr/bin/env python
from __future__ import print_function
"""EM Examples

EM examples is a set up utility codes and jupyter notebooks
that accompany the web-resource http://em.geosci.xyz
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
    name = 'em_examples',
    version = '0.0.11',
    packages = find_packages(),
    install_requires = [
        'future',
        'numpy>=1.7',
        'scipy>=0.13',
        'matplotlib',
        'Pillow',
        'requests',
        'ipywidgets',
        'properties[math]',
        'SimPEG>=0.4.1',
        'jupyter',
    ],
    author = 'Lindsey Heagy',
    author_email = 'lheagy@eos.ubc.ca',
    description = 'em_examples',
    long_description = LONG_DESCRIPTION,
    keywords = 'geophysics, electromagnetics',
    url = 'http://em.geosci.xyz',
    download_url = 'https://github.com/geoscixyz/em_examples',
    classifiers=CLASSIFIERS,
    platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License',
    use_2to3 = False,
)
