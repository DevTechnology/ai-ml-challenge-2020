# -*- coding: utf-8 -*-

import setuptools
import os
from io import open

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
  readme_contents = f.read()

setuptools.setup(
  name='dteulaml',
  version='0.0.1',
  description='dteulaml library',
  long_description=readme_contents,
  long_description_content_type='text/x-rst',
  author='Joshua Powers',
  author_email='joshua.powers@devtechnology.com',
  install_requires=['numpy', 'pandas', 'nltk', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'transformers', 'tensorboardx', 'simpletransformers', 'matplotlib'],
  dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Customers',
    'Topic :: Analytics',
    'Programming Language :: Python :: 3.7'
  ],
  keywords='analytics eula',
  packages=setuptools.find_packages(),
  package_data={'':['resources/*.csv']},
  include_package_data=True,
  python_requires='>=3.7',
  entry_points = {'console_scripts': ['dteulaml=dteulaml.__main__:main']}
)
