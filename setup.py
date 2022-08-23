# Based on the following article:
# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56

from setuptools import setup, find_packages
setup(
  name = 'HDRutils',
  packages = find_packages(),
  include_package_data=True,
  version = '0.10',
  license='MIT',
  description = 'Utility functions for performing basic operations on HDR images, including ' \
                'merging and deghosting',
  author = 'Param Hanji',
  author_email = 'param.hanji@gmail.com',
  url = 'https://github.com/gfxdisp/HDRutils',
  download_url = 'https://github.com/gfxdisp/HDRutils/archive/v0.10.tar.gz',
  keywords = ['HDR', 'Merging', 'Deghosting', 'simulation'],
  install_requires=[
          'numpy',
          'imageio==2.9.0',
          'rawpy',
          'exifread',
          'tqdm',
          'colour-demosaicing',
          'matplotlib',
          'opencv_python',
          'scipy==1.7.1',
          'scikit-image==0.18.3'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'],
)
