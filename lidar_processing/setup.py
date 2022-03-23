#!/usr/bin/env python

from setuptools import setup
import os
import re
import io

# Read the long description from the readme file
with open("README.rst", "rb") as f:
    long_description = f.read().decode("utf-8")


# Read the version parameters from the __init__.py file. In this way
# we keep the version information in a single place.
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Run setup
setup(name='lidar_processing',
      packages=['lidar_processing', 'lidar_processing.tests',],
      version=find_version("lidar_processing", "__init__.py"),
      description='Routines for atmospheric lidar processing.',
      long_description=long_description,
      url='https://bitbucket.org/iannis_b/lidar_processing/',
      author='Ioannis Binietoglou',
      author_email='ioannis@inoe.ro',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
      ],
      keywords='lidar aerosol processing',
      install_requires=[
          "netCDF4",
          "numpy",
          "matplotlib",
          "scipy",
          "sphinx",
          "numpydoc",
          "pyyaml",
      ],
      #entry_points={
      #    'console_scripts': ['licel2scc = atmospheric_lidar.scripts.licel2scc:main',
      #                        'licel2scc-depol = atmospheric_lidar.scripts.licel2scc_depol:main'],
      #},
      )

