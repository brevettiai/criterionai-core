import os
import unittest
from copy import deepcopy

from criterion_core.config import CriterionConfig
from criterion_core.utils import path


class TestCriterionConfig(unittest.TestCase):
    example_schema = {'fields': [
        {'label': 'Data', 'type': 'label'},
        {'min': 96, 'model': 'input_width', 'max': 4096, 'label': 'Network input width',
         'inputType': 'number', 'type': 'input', 'default': 224, 'required': True},
        {'min': 96, 'model': 'input_height', 'max': 4096, 'label': 'Network input height',
         'inputType': 'number', 'type': 'input', 'default': 224, 'required': True},
        {'min': 1, 'model': 'input_channels', 'max': 4, 'label': 'Image channels',
         'inputType': 'number', 'type': 'input', 'default': 3, 'required': True},
        {'label': "Output classes, separated by ','", 'inputType': 'text', 'type': 'input',
         'model': 'classes', 'required': False},
        {'hint': 'Max 5000 characters', 'required': True,
         'default': '{\n "gode": "A",\n "chips": ["B","C"]\n}', 'rows': 4, 'max': 5000,
         'label': 'Json class mapping', 'type': 'textArea', 'model': 'class_mapping',
         'placeholder': 'Map classes to folder names'}, {'label': 'Training', 'type': 'label'},
        {'min': 0, 'model': 'epochs', 'max': 50, 'label': 'Epochs', 'inputType': 'number',
         'type': 'input', 'default': 10, 'required': True},
        {'min': 1, 'model': 'batch_size', 'max': 64, 'label': 'Batch Size',
         'inputType': 'number', 'type': 'input', 'default': 32, 'required': True},
        {'model': 'loss', 'required': True, 'label': 'Loss function', 'type': 'select',
         'default': 'weighted_binary_crossentropy',
         'values': ['weighted_binary_crossentropy', 'binary_crossentropy']},
        {'hint': 'Max 5000 characters', 'default': '{\n  "chips": 10\n}', 'rows': 4,
         'max': 5000, 'label': 'Json class weights', 'type': 'textArea',
         'model': 'class_weights', 'placeholder': 'Weight some classes more than others'},
        {'label': 'Validation', 'type': 'label'},
        {'model': 'validation.split_strategy', 'required': True, 'label': 'Splitting strategy',
         'type': 'select', 'default': 'by_class', 'values': ['by_class']},
        {'min': 0, 'model': 'validation.split_percentage', 'max': 100,
         'label': 'Split percentage', 'inputType': 'number', 'type': 'input', 'default': 20,
         'required': True}, {'label': 'Data Augmentation', 'type': 'label'},
        {'min': 0, 'model': 'data_augmentation.vertical_translation_range', 'max': 100,
         'label': 'Vertical translation range in percent', 'inputType': 'number',
         'type': 'input', 'default': 1, 'required': True},
        {'min': 0, 'model': 'data_augmentation.horizontal_translation_range', 'max': 100,
         'label': 'Horizontal translation range in percent', 'inputType': 'number',
         'type': 'input', 'default': 1, 'required': True},
        {'min': 0, 'model': 'data_augmentation.scaling_range', 'max': 100,
         'label': 'Scaling range in percent', 'inputType': 'number', 'type': 'input',
         'default': 0, 'required': True},
        {'min': 0, 'model': 'data_augmentation.rotation_angle', 'max': 180,
         'label': 'Rotation range in degrees', 'inputType': 'number', 'type': 'input',
         'default': 5, 'required': True},
        {'step': 0.1, 'min': 0, 'model': 'data_augmentation.shear_range', 'max': 1,
         'label': 'Shear range', 'inputType': 'number', 'type': 'input', 'default': 0,
         'required': True},
        {'default': False, 'label': 'Can images be flipped horizontally?', 'type': 'checkbox',
         'model': 'data_augmentation.flip_horizontal', 'required': True},
        {'default': False, 'label': 'Can images be flipped vertically?', 'type': 'checkbox',
         'model': 'data_augmentation.flip_vertical', 'required': True},
        {'default': True, 'label': 'Color correction', 'type': 'checkbox',
         'model': 'data_augmentation.color_correct'}, {'label': 'Export', 'type': 'label'},
        {'model': 'img_format', 'required': True, 'label': 'Input image format in production',
         'type': 'select', 'default': 'bmp', 'values': ['bmp', 'jpeg', 'png']}
    ]}

    example_parameters = {'settings': {'loss': 'weighted_binary_crossentropy',
                                       'data_augmentation': {'flip_horizontal': True, 'rotation_angle': 5,
                                                             'color_correct': True, 'horizontal_translation_range': 1,
                                                             'flip_vertical': False, 'scaling_range': 0,
                                                             'shear_range': 0, 'vertical_translation_range': 1},
                                       'input_height': 224, 'class_mapping': '',
                                       'validation': {'split_strategy': 'by_class', 'split_percentage': 20},
                                       'batch_size': 10, 'img_format': 'bmp', 'class_weights': '', 'input_channels': 3,
                                       'epochs': 1, 'classes': 'A B', 'input_width': 224},
                          'complete_url': '/api/models/c30facd2-e0e7-4b6d-9339-14a1d22bbd54/complete?key=2FJpSa10x7x1odb07anQbxZY&modelPath=',
                          'name': 'Test Py3.5 image classification',
                          'remote_url': '/api/models/c30facd2-e0e7-4b6d-9339-14a1d22bbd54/remotemonitor?key=2FJpSa10x7x1odb07anQbxZY',
                          'api_key': '2FJpSa10x7x1odb07anQbxZY', 'host_name': 'https://app.criterion.ai',
                          'id': 'c30facd2-e0e7-4b6d-9339-14a1d22bbd54',
                          'charts_url': '/api/models/c30facd2-e0e7-4b6d-9339-14a1d22bbd54/charts?key=2FJpSa10x7x1odb07anQbxZY',
                          'datasets': [{
                              'bucket': 'C:\\data\\novo\\images\\ace29895-6bb6-49a4-a8a6-4bf47604a0b9.datasets.criterion.ai',
                              'name': 'NeurIPS vials CnD', 'id': '22df6831-5de9-4545-af17-cdfe2e8b2049'}, {
                              'bucket': 'C:\\data\\novo\\images\\555bdea5-8b57-4352-beab-fc0006232d21.datasets.criterion.ai',
                              'name': 'Data', 'id': '555bdea5-8b57-4352-beab-fc0006232d21'}]}

    def test_load_schema(self):
        config = CriterionConfig(job_dir="", schema=self.example_schema, **self.example_parameters)

    def test_load_schema_missing_param(self):
        param = deepcopy(self.example_parameters)
        del param['settings']['validation']
        with self.assertRaises(Exception) as context:
            config = CriterionConfig(job_dir="", schema=self.example_schema, **param)

        self.assertTrue("not in settings" in context.exception.args[0], msg="Settings error not raised")

    def test_get_paths(self):
        config = CriterionConfig(job_dir="", schema=self.example_schema, **self.example_parameters)
        paths = ["models", "cp"]
        self.assertEqual(config.temp_path(*paths), os.path.join(config._temporary_path.name, *paths))

        self.assertEqual(config.artifact_path(*paths), path.join(config.job_dir, "artifacts", *paths))
        config._temporary_path.cleanup()


if __name__ == '__main__':
    unittest.main()
