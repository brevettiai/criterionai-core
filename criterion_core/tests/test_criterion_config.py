import os
import unittest
from copy import deepcopy

from criterion_core.config import CriterionConfig
from criterion_core.utils import path
from criterion_core.utils.vue_schema_generator import get_default_schema


class TestCriterionConfig(unittest.TestCase):
    example_schema = get_default_schema()

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


    def test_default_schema_class_mapping_and_classes(self):
        assert any([x for x in self.example_schema["fields"] if x.get("model", "") == "class_mapping"])
        assert any([x for x in self.example_schema["fields"] if x.get("model", "") == "classes"])


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
