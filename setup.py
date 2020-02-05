import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '0.0.1'

# allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-pose-producer',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    description='pose producer',
    long_description='pose producer',
    url='https://github.com/WildflowerSchools/wf-pose-producer',
    author='meganlkm',
    entry_points={
        'console_scripts': [
            'producer=producer:main',
        ],
    }
)
