#! /usr/bin/env python

from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='deflex',
    version='0.0.1',
    author='Uwe Krien',
    author_email='uwe.krien@rl-institut.de',
    description='A reegis model of Germany with flexible multi regions.',
    namespace_package=['deflex'],
    long_description=read('README.rst'),
    packages=find_packages(),
    package_dir={'deflex': 'deflex'},
    install_requires=[
        'oemof >= 0.1.0',
        'pandas >= 0.17.0',
        'demandlib',
        'workalendar',
        'networkx',
        'numpy',
        'shapely',
        'rtree',
        'xlrd',
        'xlwt'])
