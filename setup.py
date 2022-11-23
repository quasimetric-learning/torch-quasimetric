#!/usr/bin/env python
import os
import re
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
readme = open(os.path.join(PROJECT_ROOT, 'README.md')).read()

def get_version(*path):
    version_file = os.path.join(*path)
    lines = open(version_file, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (version_file,))



setup(
    # Metadata
    name='torchqmet',
    author='Tongzhou Wang',
    author_email='tongzhou.wang.1994@gmail.com',
    url='https://github.com/quasimetric-learning/torch-quasimetric',
    install_requires=["torch>=1.11.0"],
    python_requires=">=3.7.0",
    description='PyTorch Package for Quasimetric Learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),
    version=get_version(PROJECT_ROOT, "torchqmet", "__init__.py"),

    zip_safe=True,
)
