# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['rwlenspy']

package_data = \
{'': ['*']}

install_requires = \
['astropy>=5.2.1,<6.0.0',
 'cython>=0.29.33,<0.30.0',
 'numpy>=1.24.1,<2.0.0',
 'scipy>=1.10.0,<2.0.0']

setup_kwargs = {
    'name': 'rwlenspy',
    'version': '0.1.0',
    'description': '',
    'long_description': '',
    'author': 'Zarif Kader',
    'author_email': 'kader.zarif@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.12',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
