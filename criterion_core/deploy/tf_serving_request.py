# Utility tool for getting a prediction

import argparse
import json
import logging
import time
from base64 import b64encode
from tkinter import filedialog

import requests

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='Tensorflow serving ip', default="localhost")
    parser.add_argument('--port', help='Tensorflow serving port', default=8501)
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--version', help='model version', required=False)
    parser.add_argument('--image', help='path to image', required=False)
    parser.add_argument('--repeat', help='repeat prediction n times', type=int, default=1)
    parser.add_argument('--token', help='Bearer token', default='')

    args, unparsed = parser.parse_known_args()
    return args, unparsed


if __name__ == "__main__":
    args, _ = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.version is not None:
        url = "http://{ip}:{port}/v1/models/{model}/versions/{version}:predict".format(**args.__dict__)
    else:
        url = "http://{ip}:{port}/v1/models/{model}".format(**args.__dict__)

    n_left = args.repeat
    image = args.image

    if image is None:
        image = filedialog.askopenfilename(initialdir="/", title='Please select an image')

    headers = {
        "Content-Type": "application/json"
    }
    if args.token:
        headers["Authorization"] = "Bearer {}".format(args.token)

    with open(image, "rb") as fp:
        log.info("{} -> {}".format(image, url))
        payload = {"instances": [{"image_bytes": {"b64": b64encode(fp.read()).decode("utf-8")}}]}
        ts = []
        while n_left != 0:
            tstart = time.time()
            response = requests.post(url, json=payload, headers=headers)
            tend = time.time()
            log.info(json.dumps(response.json(), indent=2))
            log.info("Prediction time: %1.1fms" % (1000 * (tend - tstart)))
            n_left -= 1
