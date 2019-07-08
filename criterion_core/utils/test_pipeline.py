from criterion_core.config import CriterionConfig

from criterion_core.folder_annotated_dataset import load_image_datasets
from criterion_core.utils.vegalite_charts import dataset_summary, make_security_selection
from criterion_core.utils.sampletools import flatten_dataset
from criterion_core.utils import data_generator
from criterion_core.utils import io_tools

from criterion_core.utils import evaluation
from criterion_core.utils import path
from criterion_core.utils.facets_atlas import build_facets

from tensorflow import keras

import numpy as np
import json
import ast
import logging
import datetime
import argparse
import json
import urllib

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', default='/output/')
    parser.add_argument('--model_id', help='Assigned id to model', default='-1')
    parser.add_argument('--info_file', type=str, help='Info file with test job info', required=True)
    args, unparsed = parser.parse_known_args()
    return args, unparsed


def get_config(schema_path):
    args, _ = parse_args()

    training_info = json.loads(io_tools.read_file(path.join(args.job_dir, "info.json")).decode("utf-8"))
    training_info["settings"]["max_epoch_samples"] = np.inf # hack: Using setting to allow limited samples for debugging
    config = CriterionConfig.from_mlengine(args.job_dir, schema_path, info_file=args.info_file,
                                           check_required_settings=False, settings_overload=training_info["settings"])
    return config


def setup_data_generators(config):
    class_mapping = ast.literal_eval(config.settings.class_mapping.strip(' ')) if len(
        config.settings.class_mapping) > 0 else {}
    ds_samples = load_image_datasets(config.datasets, class_map=class_mapping, force_categories=False)

    # Split data
    test_set = list(flatten_dataset(ds_samples))

    config.upload_chart("Test set", dataset_summary(test_set))

    input_shape = (config.settings.input_height, config.settings.input_width)
    rois = json.loads(config.settings.rois)

    class_path = config.artifact_path("classes.json")
    classes = json.loads(io_tools.read_file(class_path).decode("utf-8"))

    test_gen = data_generator.DataGenerator(test_set, classes=classes,
                                            batch_size=config.settings.batch_size, augmentation=None,
                                            target_shape=input_shape, max_epoch_samples=config.settings.max_epoch_samples,
                                            color_mode=config.settings.color_mode, shuffle=True, rois=rois)
    return test_gen, classes, class_mapping, rois, input_shape, test_set


def evaluate_model(config, finetuned_model, test_gen, rois, classes, max_facet_size=4000, **kwargs):
    test_pred_output = evaluation.sample_predictions(finetuned_model, test_gen)
    config.upload_artifact('test_data_{}_full.csv'.format(config.id), test_pred_output.to_csv(sep=';', encoding='utf-8'))
    summary, tag_fields = evaluation.pivot_summarizer(test_pred_output, config.datasets, config.tags, classes, **kwargs)
    config.upload_pivot_data(summary, tag_fields)

    for cl, chart in zip(classes, make_security_selection(test_pred_output, classes)):
        config.upload_chart("Set {} security levels".format(cl), chart)

    log.info("Building Facets for developmentset")
    facet_key = '_{}_{}'.format(config.id, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    facet_gen = data_generator.DataGenerator(test_pred_output[:max_facet_size].to_dict("records"), augmentation=None,
                                             rois=rois, target_shape=(64, 64), shuffle=False,
                                             interpolation='linear', anti_aliasing=0.5,
                                             color_mode=config.settings.color_mode, rescale=255.0, offset=0.0)
    facets_dive, facets_img = build_facets(facet_gen, config.get_facets_folder(), facet_key=facet_key)
    facet_info = '&' + urllib.parse.urlencode({'facetsFile': facets_dive, 'spriteFile': facets_img})
    return facet_info


def load_model(config, model_path, **kwargs):
    cp_model_path = config.temp_path("model_cp.h5")
    with open(cp_model_path, 'wb') as model_file:
        model_file.write(io_tools.read_file(model_path))
    model = keras.models.load_model(cp_model_path, **kwargs)
    return model