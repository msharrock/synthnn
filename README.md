synthnn
=======================

[![Build Status](https://travis-ci.org/jcreinhold/synthnn.svg?branch=master)](https://travis-ci.org/jcreinhold/synthnn)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/synthnn/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/synthnn?branch=master)
[![Documentation Status](https://readthedocs.org/projects/synthnn/badge/?version=latest)](http://synthnn.readthedocs.io/en/latest/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package contains deep neural network-based (pytorch) modules to synthesize magnetic resonance (MR) and computed 
tomography (CT) brain images. Synthesis is the procedure of learning the transformation that takes a specific contrast image to another estimate contrast.

For example, given a set of T1-weighted (T1-w) and T2-weighted (T2-w) images, we can learn the function that maps the intensities of the
T1-w image to match that of the T2-w image via a UNet or other deep neural network architecture. In this package, we supply 
the framework and several models for this type of synthesis. See the `Relevant Papers` section (at the bottom of 
the README) for a non-exhaustive list of some papers relevant to the work in this package.

We also support a *non*-DNN-based synthesis package called [synthit](https://gitlab.com/jcreinhold/synthit).

** Note that this is an **alpha** release. If you have feedback or problems, please submit an issue (it is very appreciated) **

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/synthnn)

Requirements
------------

- nibabel
- numpy
- torch
- torchvision

Installation
------------

    pip install git+git://github.com/jcreinhold/synthnn.git

Tutorial
--------

[5 minute Overview](https://github.com/jcreinhold/synthnn/blob/master/tutorials/5min-tutorial.ipynb)

In addition to the above small tutorial, there is consolidated documentation [here](https://synthnn.readthedocs.io/en/latest/).

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

Relevant Papers
---------------

[1] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling from CT Using Synthetic MR Images,” in MICCAI MLMI, vol. 10541, pp. 291–298, 2017.