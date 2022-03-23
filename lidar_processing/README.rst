.. image:: https://readthedocs.org/projects/lidar-processing/badge/?version=latest
   :target: http://lidar-processing.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codeship.com/projects/2e21b760-6eaf-0134-9495-3e75f4fffff1/status?branch=default
   :target: https://codeship.com/projects/177870
   :alt: Build Status

Description
===========

This module collects basic processing routines for aerosol lidar systems.

The module should include only the pre-processing and optical processing functions. Reading data, visualization, etc.
should be handled by different modules.



Installation
------------

The module is tested for Python 2.7.* and slightly for Python 3.6

The suggested method to install is to clone the repository and install it using the -e command.

.. sourcecode:: console

   pip install -e ./lidar_processing

assuming that the module is cloned in the lidar_processing directory.

The installation procedure is not yet fully automatic. You may need to install numpy, scipy manually. Probably
the best way to install numpy and scipy is through a distribution like `anaconda <https://www.continuum.io/downloads>`_.

You will also need to install the `lidar_molecular <https://bitbucket.org/iannis_b/lidar_molecular>`_ module. You
can do this by

.. sourcecode:: console

   pip install -r requirements.txt

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


Notebooks and data
------------------
The project includes some test data in the /data/ folder. It also includes some ipython notebooks with some
example processing of the data. You can run the notebook with the command:

.. sourcecode:: console

   jupyter notebook


Sponsors
--------
The development of this module is supported by `Raymetrics S.A. <https://www.raymetrics.com/>`_.

.. image:: logos/raymetrics_logo.png
   :target: https://www.raymetrics.com/
   :alt: Raymetrics logo


