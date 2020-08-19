# -*- coding: utf-8 -*-

import setuptools
import os
from io import open

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
  readme_contents = f.read()

setuptools.setup(
  name='dteulaapp',
  version='0.0.1',
  description='dteulaapp library',
  long_description=readme_contents,
  long_description_content_type='text/x-rst',
  author='Joshua Powers',
  author_email='joshua.powers@devtechnology.com',
  install_requires=['flask', 'PyPDF2', 'python-docx', 'numpy', 'pandas', 'nltk', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'transformers', 'tensorboardx', 'simpletransformers'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Customers',
    'Topic :: Analytics',
    'Programming Language :: Python :: 3.7'
  ],
  keywords='analytics eula',
  packages=setuptools.find_packages(),
  package_data={'':['resources/*.csv', 'resources/*.docx', 'resources/*.pdf']},
  include_package_data=True,
  python_requires='>=3.7',
  entry_points = {'console_scripts': ['dteulaapp=dteulaapp.__main__:main']}
)
