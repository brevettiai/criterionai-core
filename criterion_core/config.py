import json
from .utils import path
import logging
import requests
from itertools import chain
from collections import defaultdict
from types import SimpleNamespace
from json import JSONEncoder
import tempfile
from .utils import io_tools

log = logging.getLogger(__name__)


class CriterionConfig:
    """
    Interface for reading configurations from criterion.ai
    """

    def __init__(self, job_dir, id, name, datasets, settings, api_key, host_name, charts_url, complete_url, remote_url,
                 schema, check_required_settings=True, root_tags=None, **kwargs):
        """
        :param schema: JSON schema used for model definition
        :param parameters: setup parameters overwriting default schema values
        """
        self.job_dir = job_dir
        self.id = id
        self.name = name
        self.datasets = datasets
        self.settings = merge_settings(schema, settings, check_required_settings)
        self.tags = root_tags

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

        parameter_path = path.join(job_dir, "info.json")
        log.info("Config args found at: '%s'" % parameter_path)
        parameters = json.loads(str(io_tools.read_file(parameter_path), "utf-8"))
        log.info(parameters)

        with open(schema_path, 'r') as fp:
            schema = json.load(fp)

        config = CriterionConfig(job_dir=job_dir, schema=schema, **parameters)
        log.info(config)
        return config

    def get_facets_folder(self):
        return self.job_dir

    def temp_path(self, *paths):
        io_tools.make_dirs(self._temporary_path.name)
        return path.join(self._temporary_path.name, *paths)

    def artifact_path(self, *paths):
        io_tools.make_dirs(path.join(self.job_dir, "artifacts", *paths[:-1]))
        return path.join(self.job_dir, "artifacts", *paths)

    def upload_pivot_data(self, summary, tag_fields, summary_path=None, fields_path=None):
        if summary_path is None:
            summary_path = self.artifact_path("pivot", "classification_summary_{}.json".format(self.id))

        if fields_path is None:
            fields_path = self.artifact_path("pivot", "summary_fields_{}.json".format(self.id) )


        log.info("Uploading summary to {}".format(summary_path))
        io_tools.write_file(summary_path, json.dumps(summary))

        log.info("Uploading summary fields to {}".format(fields_path))

        io_tools.write_file(fields_path,
                            json.dumps(tag_fields))

    def upload_chart(self, name, vegalite_json):
        charts_url = '{}{}&name={}'.format(self.host_name, self.api_endpoints['charts'], name)
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

        gcs_model_path = path.join(self.job_dir, package_path)
        log.info("Saving saved_model to {}".format(gcs_model_path))
        with open(tmp_package_path, 'rb') as f_package:
            io_tools.write_file(gcs_model_path, f_package.read())

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
        return self.ConfigEncoder().encode(self)

    class ConfigEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, CriterionConfig):
                return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            return o.__dict__


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
        if uri[0] not in d:
            return False
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
