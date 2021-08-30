"""

Build and install the project: geoscilabs

geoscilabs are the codes that support the interactive labs provided through
https://gpg.geosci.xyz and https://em.geosci.xyz
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="geoscilabs",
    version="0.3.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "ipywidgets",
        "SimPEG>=0.15.0",
        "discretize>=0.7.0",
        "empymod>=2.0.0",
        "jupyter",
        "deepdish",
        "Pillow",
        "requests",
        "cvxopt",
    ],
    author="GeoSci Developers",
    author_email="lindseyheagy@gmail.com",
    description="geoscilabs",
    long_description=LONG_DESCRIPTION,
    keywords="geophysics, electromagnetics",
    url="https://geosci.xyz",
    download_url="https://github.com/geoscixyz/geosci-labs",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="MIT License",
    use_2to3=False,
)
