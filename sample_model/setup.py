from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras==2.2.4',
                     'h5py==2.8.0',
                     'Pillow==5.2.0',
                     'scikit-learn',
                     'altair @ git+https://github.com/criterion-ai/altair.git/#egg=master-3.0.0dev2',
                     'criterion_core @ git+https://github.com/criterion-ai/criterionai-core.git/@v0.1#egg=criterion_core',
                     'google-cloud-storage==1.14.0',
                     'scipy==1.2.0',
                     'imgaug==0.2.8']

setup(
    name='criterion-packages-image_segmentation',
    version='0.2',
    url='https://github.com/mypackage.git',
    author='Emil Tyge',
    author_email='emil@criterion.ai',
    packages=find_packages(),
    description='Image segmentation model training for criterion.ai',
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)