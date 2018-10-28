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

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name = 'em_examples',
    version = '0.0.33',
    packages = find_packages(),
    install_requires = [
        'future',
        'numpy>=1.7',
        'scipy>=0.13',
        'matplotlib>2.1.',
        'Pillow',
        'requests',
        'ipywidgets',
        'SimPEG>=0.4.1',
        'jupyter',
        'empymod',
        'deepdish',
        'pymatsolver>=0.1.2'
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
