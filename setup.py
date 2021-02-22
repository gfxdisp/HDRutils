from distutils.core import setup
setup(
  name = 'HDRutils',
  packages = ['HDRutils'],
  version = '0.1',
  license='MIT',
  description = 'Utility functions for perfming basic operations on HDR images, including merging and deghosting',
  author = 'Param Hanji',
  author_email = 'param.hanji@gmail.com',
  url = 'https://github.com/catchchaos/HDRutils-pip',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz', # TODO: https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
  keywords = ['HDR', 'Merging', 'Deghosting'],
  install_requires=[            # I get to this in a second
          'numpy',
          'imageio',
          'rawpy',
          'exifread',
          'tqdm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)