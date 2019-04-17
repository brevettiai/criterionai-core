import json
from .utils import path  # , gcs_config_hacks
import logging
import requests
from itertools import chain
from collections import defaultdict
from types import SimpleNamespace
import os
import tempfile
from fs import open_fs

log = logging.getLogger(__name__)


class CriterionConfig:
    """
    Interface for reading configurations from criterion.ai
    """

    def __init__(self, job_dir, id, name, datasets, settings, api_key, host_name, charts_url, complete_url, remote_url,
                 schema, check_required_settings=True, **kwargs):
        """
        :param schema: JSON schema used for model definition
        :param parameters: setup parameters overwriting default schema values
        """
        self.job_dir = job_dir
        self.id = id
        self.name = name
        self.datasets = datasets
        self.settings = merge_settings(schema, settings, check_required_settings)

        self.api_key = api_key
        self.host_name = host_name
        self.api_endpoints = dict(
            charts=charts_url,
            complete=complete_url,
            remote=remote_url,
        )
        self._temporary_path = tempfile.TemporaryDirectory(prefix=self.name + "-")
        self.__dict__.update(kwargs)

    @staticmethod
    def from_mlengine(job_dir, schema_path):
        """
        Get CriterionConfig from ml engine and local schema
        :param job_dir: google storage bucket of job
        :param schema_path: path for model schema
        :return: CriterionConfig object
        """

        job_fs = open_fs(job_dir)
        
        parameter_path = "/info.json"

        log.info(parameter_path)
        with job_fs.open(parameter_path, 'r') as fp:
            parameters = json.load(fp)

        log.info(parameters)
        log.info(schema_path)

        log.info("Find schema")
        log.info(os.path.dirname(os.path.realpath(__file__)))

        log.info(os.getcwd())
        with open(schema_path, 'r') as fp:
            schema = json.load(fp)

        log.info(schema)
        return CriterionConfig(job_dir=job_dir, schema=schema, **parameters)

    def get_facets_folder(self):
        return self.job_dir

    def temp_path(self, *paths):
        return path.join(self._temporary_path, *paths)

    def artifact_path(self, *paths):
        return path.join(self.job_dir, "artifacts", *paths)

    def upload_chart(self, name, vegalite_json):
        charts_url = '{}{}&name={}'.format(self.host_name + self.api_endpoints['charts'] + name)
        try:
            r = requests.post(charts_url, headers={'Content-Type': 'application/json'}, data=vegalite_json)
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP error on complete job", exc_info=e)
        except requests.exceptions.RequestException as e:
            log.warning("No Response on complete job", exc_info=e)
        return r

    def complete_job(self, tmp_package_path, package_path="artifacts/saved_model.tar.gz"):
        """
        Complete job by uploading package to gcs and notifying api
        :param tmp_package_path: Path to tar archive with python package
        :param package_path: package path on gcs
        :return:
        """

        gcs_model_path = self.job_dir + "/" + package_path
        file_io.copy(tmp_package_path, gcs_model_path, overwrite=True)

        complete_url = self.host_name + self.api_endpoints['complete'] + package_path
        try:
            r = requests.post(complete_url)
            log.info('Job completed')
            return r
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP error on complete job", exc_info=e)
        except requests.exceptions.RequestException as e:
            log.warning("No Response on complete job", exc_info=e)

        self._temporary_path.cleanup()

    def __str__(self):
        raise NotImplementedError


def merge_settings(schema, settings, check_required_fields=True):
    """
    Merge schema and settings
    :param schema:
    :param settings:
    :param check_required_fields: enforce required fields in schema exists in settings
    :return:
    """
    fields = schema['fields']
    fields.extend(chain(g['fields'] for g in schema.get('groups', [])))
    fields = [f for f in fields if "model" in f]

    default_settings = defaultdict(defaultdict)
    for f in fields:
        uri = f['model'].split('.')
        s = default_settings
        if len(uri) > 1:
            for key in uri[:-1]:
                s = s[key]
        s[uri[-1]] = f.get('default')
        if check_required_fields and f.get('required', False):
            assert in_dicts(settings, uri), "%r not in settings" % uri

    dict_merger(settings, default_settings)

    return dicts_to_simple_namespace(default_settings)


def dict_merger(source, target):
    """
    Merge two dicts of dicts
    :param source:
    :param target:
    :return:
    """
    for k, v in source.items():
        if isinstance(v, dict):
            dict_merger(v, target[k])
        else:
            target[k] = v


def in_dicts(d, uri):
    """
    Check if path of keys in dict of dicts
    :param d: dict of dicts
    :param uri: list of keys
    :return:
    """
    if len(uri) > 1:
        return in_dicts(d[uri[0]], uri[1:])
    else:
        return uri[0] in d


def dicts_to_simple_namespace(dicts):
    """
    Transform dicts of dicts to tree of SimpleNamespaces
    :param dicts:
    :return:
    """
    for k, v in dicts.items():
        if isinstance(v, dict):
            dicts[k] = dicts_to_simple_namespace(v)
    return SimpleNamespace(**dicts)


if __name__ == '__main__':
    schemafile = r'C:\Users\emtyg\dev\novo\criterionai-packages\imageclassification\settings-schema.json'
    with file_io.FileIO(schemafile) as fp:
        schema = json.load(fp)

    CriterionConfig("", "", "", {}, "", "", "", schema)
