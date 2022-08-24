# Description:
#  
# Written by Ruiming Cao on January 31, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  with open(os.path.join(_CURRENT_DIR, 'calcil', '__init__.py')) as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `calcil/__init__.py`')

install_requires = [
]

setup(
    name='calcil',
    version=_get_version(),
    url='https://github.com/rmcao/CalCIL',
    license='BSD',
    author='Ruiming Cao',
    description='Berkeley Computational Imaging Library',
    long_description=open(os.path.join(_CURRENT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    author_email='rcao@berkeley.edu',
    keywords='computational imaging',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=install_requires,
    python_requires='>=3.7',
)
