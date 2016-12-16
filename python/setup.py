#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Datajoint schemata and analysis code for the mouse pipeline."


setup(
    name='pipeline',
    version='0.1.0.dev1',
    description="data processing chain for MICrONS project team ninai",
    long_description=long_description,
    author='Fabian Sinz, Dimitri Yatsenko',
    author_email='sinz@bcm.edu',
    license="GNU LGPL",
    url='https://github.com/cajal/pipeline',
    keywords='neuroscientific data processing',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy','sh','matplotlib','pandas','seaborn','scipy','imageio','pyfnnd'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: GNU LGPL',
        'Topic :: Database :: Front-Ends',
    ],
    scripts=[
        'scripts/worker',
        'scripts/worker-populate.py',
        'scripts/worker-environment.py',
    ]
)

RUN conda install -c menpo opencv3=3.1.0
RUN conda install -c https://conda.anaconda.org/omnia cvxpy
# RUN git clone --recursive -b agiovann-master https://github.com/valentina-s/Constrained_NMF.git
# RUN git clone --recursive https://github.com/agiovann/Constrained_NMF.git
RUN git clone --recursive -b dev https://github.com/agiovann/Constrained_NMF.git
WORKDIR /Constrained_NMF/
RUN conda install --file requirements_conda.txt
RUN pip install -r requirements_pip.txt
RUN apt-get install libc6-i386
RUN apt-get install -y libsm6 libxrender1
RUN conda install pyqt=4.11.4
RUN python setup.py install
