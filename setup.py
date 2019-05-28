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
    'gcloud-aio-auth @ git+https://github.com/elempollon/gcloud-aio.git/#egg=gcloud-aio-auth-2.0.1&subdirectory=auth',
    'gcloud-aio-storage @ git+https://github.com/elempollon/gcloud-aio.git/#egg=gcloud-aio-storage-4.1.0&subdirectory=storage',
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
