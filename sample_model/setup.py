import os
from setuptools import find_packages
from setuptools import setup

schema_path = "model/settings-schema.json"
try:
    from criterion_core.utils.vue_schema_generator import generate_model_parameter_schema

    generate_model_parameter_schema(path=schema_path)
except ImportError:
    if not os.path.isfile(schema_path) or not os.path.isfile("MANIFEST.in"):
        raise FileNotFoundError("Settings schema and package manifest " +
                                "must exist at %r and 'MANIFEST.in" % schema_path)

REQUIRED_PACKAGES = ['keras==2.2.4',
                     'h5py==2.8.0',
                     'Pillow==5.2.0',
                     'scikit-learn',
                     'altair @ git+https://github.com/criterion-ai/altair.git/#egg=master',
                     'criterion_core @ git+https://github.com/criterion-ai/criterionai-core.git/@v0.2#egg=criterion_core',
                     'google-cloud-storage==1.14.0',
                     'scipy==1.2.0']

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