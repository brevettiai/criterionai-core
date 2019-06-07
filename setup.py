from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy>=1.16',
    'pandas>=0.24',
    'requests>=2.21.0',
    'opencv-python>=4.1.0',
    'altair @ git+https://github.com/criterion-ai/altair.git/#egg=master',
    'aiofiles==0.4.0',
    'aiohttp==2.3.10',
    "aiofiles==0.4.0",
    'Pillow==5.2.0',
    'gcloud-aio-auth @ git+https://github.com/elempollon/gcloud-aio.git/#egg=master&subdirectory=auth',
    'gcloud-aio-storage @ git+https://github.com/elempollon/gcloud-aio.git/#egg=master&subdirectory=storage',
    'google-auth==1.6.3',
    'google-cloud-storage==1.16.0',
    'imgaug==0.2.9'
]

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
