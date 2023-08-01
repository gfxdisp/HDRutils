# Based on the following article:
# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
  name = 'HDRutils',
  packages = find_packages(),
  package_data = {'HDRutils': ['noise_modeling/darktable.json']},
  include_package_data=True,
  version = '1.0',
  license='MIT',
  description = 'Utility functions for performing basic operations on HDR images, including ' \
                'merging and deghosting',
  author = 'Param Hanji',
  author_email = 'param.hanji@gmail.com',
  url = 'https://github.com/gfxdisp/HDRutils',
  long_description=long_description,
  long_description_content_type='text/markdown',
  keywords = ['HDR', 'Merging', 'Deghosting', 'simulation'],
  install_requires=[
          'numpy',
          'imageio>=2.21.2',
          'rawpy',
          'exifread',
          'tqdm',
          'colour-demosaicing',
          'matplotlib',
          'scipy>=1.7.1',
          'scikit-image>=0.18.3',
          'pyexr'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'],
)
