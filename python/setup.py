#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Datajoint schemata and analysis code for the mouse pipeline."


setup(
    name='pipeline',
    version='0.2.0',
    description="data processing chain for MICrONS project team ninai",
    long_description=long_description,
    author='Fabian Sinz, Dimitri Yatsenko, Erick Cobos',
    author_email='sinz@bcm.edu',
    license="GNU LGPL",
    url='https://github.com/cajal/pipeline',
    keywords='neuroscientific data processing',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
#    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
#                      'imageio', 'imreg_dft', 'slacker', 'pyfftw', 'h5py',
#                      'scikit-image', 'scikit-learn'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
                      'imageio', 'imreg_dft', 'slacker', 'h5py',
                      'scikit-image', 'scikit-learn'],

    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6 :: Only',
        'License :: OSI Approved :: GNU LGPL',
        'Topic :: Database :: Front-Ends',
    ],
    scripts=['scripts/populate-minion.py']
)

