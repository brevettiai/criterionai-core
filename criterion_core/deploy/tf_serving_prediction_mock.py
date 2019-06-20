import argparse
import json
import os
from base64 import b64decode

from flask import Flask, request, jsonify


class TFServingMock:
    """
    Tensorflow serving mock server
    Implements the tensorflow serving prediction and status apis
    """
    app = Flask(__name__)

    def __init__(self, model_config=None, monitoring_config=None, save_prediction_path=None,
                 save_prediction_name="prediction_{model}_{version}_{idx:03d}.bmp"):
        """

        :param model_config: Model config file (protobuf)
        :param monitoring_config:  Monitoring config file (protobuf) not in use
        :param save_prediction_path: Path to save inbound requests
        :param save_prediction_name: Format string for generation of filename, allowed keys are (model,version,idx)
        """
        self.model_config = model_config
        self.save_prediction_path = save_prediction_path
        self.save_prediction_name = save_prediction_name
        if save_prediction_path:
            os.makedirs(save_prediction_path, exist_ok=True)

        if model_config is not None:
            self.models = {m["name"]: m for m in model_config["model_config_list"]["config"]}
        self.build_endpoints()
        self.prediction_idx = 0

    def metrics(self):
        return """
    # HELP metric_name Description of the metric
    # TYPE metric_name type
    # Comment that's not parsed by prometheus
    http_requests_total{method="post",code="400"}  3   1391324
        """

    def status(self, model, version=None):
        if self.model_config is not None:
            if model not in self.models:
                return jsonify({"error": "Could not find any versions of model %s" % model}), 404

        model_basepath = self.models[model]["base_path"].replace("\\\\", "\\")
        versions = list(sorted((int(x) for x in os.listdir(model_basepath) if x.isdigit()), reverse=True))

        if self.model_config is not None:
            if version not in versions:
                return jsonify({"error": "Could not find version %d of model %s" % (version, model)}), 404

        status = []
        for v in versions:
            if version is None or version == v:
                status.append(dict(
                    version=v,
                    state="Available" if len(status) == 0 else "End",
                    status=dict(
                        error_code="OK",
                        error_message=""
                    ),
                ))

        return jsonify({"model_version_status": status})

    def predict(self, model, version=None):
        self.prediction_idx += 1
        length = int(request.headers.get("Content-Length", 0))

        if self.save_prediction_path is not None:
            payload = json.loads(request.data.decode("utf8")) if length > 0 else {}
            fname = os.path.join(self.save_prediction_path,
                                 self.save_prediction_name.format(idx=self.prediction_idx,
                                                                  model=model,
                                                                  version=version or ""))
            with open(fname, 'wb') as fp:
                data = b64decode(payload["instances"][0]["image_bytes"]["b64"])
                fp.write(data)

        if self.model_config is not None:
            if not model in self.models:
                return jsonify({"error": "Could not find any versions of model %s" % model}), 404

        response = {"predictions": [[1, 0, 0, 0]]}

        return jsonify(response)

    def build_endpoints(self):
        self.app.route("/v1/models/<model>")(self.status)
        self.app.route("/v1/models/<model>/versions/<int:version>")(self.status)
        self.app.route("/v1/models/<model>:predict", methods=['POST'])(self.predict)
        self.app.route("/v1/models/<model>/versions/<int:version>:predict", methods=['POST'])(self.predict)

        self.app.route("/monitoring/prometheus/metrics/")(self.metrics)


def pbParse(lines, data=None):
    """
    Mini specificationless protobuf parser
    :param lines: list of lines to parse
    :param data: dict to update with data
    :return: protobuf string parsed as dict
    """
    if data is None:
        data = {}

    while len(lines) > 0:
        line = lines[0]
        ld = line.split(" ")
        if ld[0].endswith(":"):  # Parse key/value
            ld = line.split(" ")
            if ld[0].endswith(":"):
                key = ld[0][:-1]
                if ld[1].startswith('"') and ld[1].endswith('"'):
                    value = ld[1][1:-1]
                else:
                    value = float(ld[1])
                data[key] = value

        elif ld[-1].endswith("{"):  # Parse dict
            key = lines[0].split(" ")[0]

            count = 1
            ix = 0
            while count > 0:
                ix += 1
                if lines[ix].endswith("}"):
                    count -= 1
                elif lines[ix].endswith("{"):
                    count += 1

            if key in data:
                if not isinstance(data[key], list):
                    data[key] = [data[key]]
                data[key].append(pbParse(lines[1:ix]))
            else:
                data[key] = pbParse(lines[1:ix])
            lines = lines[ix:]

        lines = lines[1:]
    return data


def pbLoads(data):
    """
    Utility function for protobuf parsing
    :param data: filecontents
    :return: protobuf string parsed as dict
    """
    lines = [x.strip() for x in data.split("\n")]
    return pbParse(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_file', help='Tensorflow serving model configuration file', required=False)
    parser.add_argument('--monitoring_config_file',
                        help='Tensorflow serving metrics congifuration file (Not implemented)', required=False)
    parser.add_argument('--output', help='Output directory if images should be logged, standard is .bmp files',
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.model_config_file:
        with open(args.model_config_file) as fp:
            model_config = fp.read()
        model_config = pbLoads(model_config)
    else:
        model_config = None

    serving = TFServingMock(model_config, save_prediction_path=args.output)
    serving.app.run(port=8501, debug=True)
