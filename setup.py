
from setuptools import setup, find_packages

setup(
    name='LFPredict',
    version='0.1',
    description='Deep learning for predicting instantaneous band-limited amplitude and phase from broadband signals.',
    author='Martin Zeller',
    author_email='martin.zeller@fau.de',
    url='https://github.com/zeller-martin/LFDeep',
    packages=find_packages(exclude=('examples')),
    install_requires=['tensorflow', 'numpy', 'scipy', 'matplotlib'],
)
