import argparse
import logging
import os

from criterion_core import CriterionConfig, load_image_datasets
from criterion_core.utils.sampletools import split_datasets

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', required=True)
    args, unparsed = parser.parse_known_args()
    return args, unparsed


def train(config):
    dssamples = load_image_datasets(config.datasets)

    # Split data
    trainingset, developmentset = split_datasets(dssamples, strategy=config.settings.validation.split_strategy,
                                                 split=config.settings.validation.split_percentage / 100)

    tmp_package_path = config.temp_path("package.tar.gz")

    # TODO Insert your training code here -> put model output at temp_package_path

    log.info("Completing job")
    config.complete_job(tmp_package_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args, _ = parse_args()

    schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings-schema.json')
    config = CriterionConfig.from_mlengine(args.job_dir, schema_path)

    log.info("Start Training")
    train(config)
