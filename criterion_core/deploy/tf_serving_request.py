# Utility tool for getting a prediction

import argparse
import json
import logging
import time
from base64 import b64encode

import requests

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='Tensorflow serving ip', default="localhost")
    parser.add_argument('--port', help='Tensorflow serving port', default=8501)
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--version', help='model version', required=False)
    parser.add_argument('--image', help='path to image', required=True)

    args, unparsed = parser.parse_known_args()
    return args, unparsed


if __name__ == "__main__":
    args, _ = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.version is not None:
        url = "http://{ip}:{port}/v1/models/{model}/versions/{version}:predict".format(**args.__dict__)
    else:
        url = "http://{ip}:{port}/v1/models/{model}:predict".format(**args.__dict__)

    image = args.image

    with open(image, "rb") as fp:
        log.info("{} -> {}".format(image, url))
        payload = {"instances": [{"image_bytes": {"b64": b64encode(fp.read()).decode("utf-8")}}]}
        ts = []
        tstart = time.time()
        response = requests.post(url, json=payload)
        tend = time.time()
        log.info(json.dumps(response.json(), indent=2))
        log.info("Prediction time: %1.1fms" % (1000 * (tend - tstart)))
