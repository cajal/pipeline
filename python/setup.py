#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Datajoint schemata and code for neuroscientific experiments in the Tolias lab."


setup(
    name='microns',
    version='0.1.0.dev1',
    description="A collection of datajoint schemas and analysis code.",
    long_description=long_description,
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License",
    url='https://github.com/atlab/pipeline',
    keywords='database organization',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    # dependency_links = ['https://github.com/datajoint/datajoint-python/tarball/master#egg=datajoint-0.1.0beta'],
    install_requires=['numpy', 'matplotlib', 'seaborn'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved ::  Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License',
        'Topic :: Database :: Front-Ends',
    ],
)
