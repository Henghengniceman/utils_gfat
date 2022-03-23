Introduction
============

This module collects basic processing routines for aerosol lidar systems. Its aim is to act as a repository
of pre-processing and optical processing routines, that can be used as a basic building block for any atmospheric
lidar processing software.

To make it reusable, the module includes only the pre-processing and optical processing functions.
Reading data, visualization, etc. should be handled by different modules.


.. note::
   Here is a list of complementary lidar-related modules:

   Molecular scattering
      The `lidar_molecular <https://bitbucket.org/iannis_b/lidar_molecular>`_  module is a collection of scripts to calculate
      scattering parameters of molecular atmosphere.

   Raw lidar files
      The `atmospheric-lidar <https://bitbucket.org/iannis_b/atmospheric-lidar>`_ module contains classes to read raw lidar
      data files, including Licel binary files. It can be used for plotting (quicklooks) and converting raw data to
      SCC format.

   EARLINET optical file reader
      The `earlinet files <https://bitbucket.org/iannis_b/earlinet-optical-file-reader>`_ files can be use to *read* and
      plot aerosol properties stored in the EARLINET netdf file format.


Installation
------------

The module is tested against Python 2.7.*

The code is developed using the mercurial version control software. The repository is hosted on
`bitbucket <https://bitbucket.org/iannis_b/lidar_processing>`_.

.. note::
   For an introduction to mercurial version control software check out `<http://hginit.com/>`_.

   If you prefer to use a program with GUI, try using `Easy mercurial <http://easyhg.org/>`_.

The suggested method to install is to clone the repository and install it using the -e command.

.. sourcecode:: console

   pip install -e ./lidar_processing

assuming that the module is cloned in the lidar_processing directory.

The installation procedure is not yet fully automatic. You will need to install manually: numpy, scipy, and
the `lidar_molecular <https://bitbucket.org/iannis_b/lidar_molecular>`_ module. The best way to install numpy and scipy
is through a distribution like `anaconda <https://www.continuum.io/downloads>`_.

When this and the *molecular* modules mature, we should optimize the installation procedure.


Documentation
-------------

Each function should be documented following the Numpy doc style.

For details see the `numpy documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.


All docstrings are collected to a single documentation file using the `Sphinx <http://www.sphinx-doc.org/>`_ module.
The documentation is located in the docs/ folder. The documentation is written in
`restructured text <http://www.sphinx-doc.org/en/stable/rest.html>`_ format.

You can rebuild the docs by running the following command from the docs folder.

.. sourcecode:: console

   make html

The documentation is also built automatically every time you push your changes to the repository. You can find it online
in `Read the docs <https://readthedocs.org/projects/lidar-processing/>`_.


Testing
-------
Some tests, based on unittest2 library, are located in the lidar_processing/tests/ folder.

You can run all the test using the commands from the project directory.

.. sourcecode:: console

   python -m unittest discover

Todo
----
The module is still in a very early stage so most things need to be done. Here is an indicative list of things to add:

* Signal gluing
* Optical product gluing (e.g. from near and far range telescopes).
* Error propagation (Monte Carlo method).
* Klett algorithm for elastic lidar retrieval
* Raman scattering algorithms for backscatter and extinction.

Even if you don't have something to code, there are other ways to contribute, e.g:

* Review/improve this documentation.
* Test that the implemented functions work correctly.
* Suggest missing routines or other improvements.



