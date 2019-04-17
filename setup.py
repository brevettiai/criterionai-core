from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud-storage==1.14.0',
                     'fs==2.4.4',
                     'fs-gcsfs==0.4.1']

setup(
    name='criterion_core',
    version='0.1',
    url='https://github.com/mypackage.git',
    author='Emil Tyge',
    author_email='emil@criterion.ai',
    packages=find_packages(),
    description='Image segmentation model training for criterion.ai',
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)
