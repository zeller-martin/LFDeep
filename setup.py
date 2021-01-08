
from setuptools import setup, find_packages

setup(
    name='deep_loop',
    version='0.1',
    description='Tools for ephys analysis',
    author='Martin Zeller',
    author_email='martin.zeller@fau.de',
    url='https://github.com/zeller-martin/deep_loop',
    packages=find_packages(exclude=('tests', 'docs'))
)
