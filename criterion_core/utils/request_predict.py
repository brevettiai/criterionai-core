import requests
from PIL import Image
import base64
from io import BytesIO
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Model URL', required=True)
    parser.add_argument('--token', help='Access token to model', required=True)
    parser.add_argument("images", nargs="+", help='Image files request predictions for')
    return parser.parse_args()


def predict_json(data, url, token):
    header = {
        "Authorization": "Bearer {}".format(token),
        "Content-Type": "application/json"
    }
    return requests.post(url, headers=header, json=data).json()


PIL_FORMATS = {
    "JPG": "JPEG"
}


def get_b64_bytes_from_path(path):
    img = Image.open(path)
    format = path.rsplit('.', 1)[-1].upper()
    buffered = BytesIO()
    img.save(buffered, format=PIL_FORMATS.get(format, format))
    return base64.b64encode(buffered.getvalue()).decode()


if __name__ == "__main__":
    args = parse_args()

    # Build request
    data = dict(instances=[dict(image_bytes=dict(b64=get_b64_bytes_from_path(p))) for p in args.images])

    # Get response
    print("Getting response for {}".format(args.images))
    prediction = predict_json(data, args.url, args.token)

    locations = [np.array(p['locations']) for p in prediction['predictions']]

    for img_path, loc in zip(args.images, locations):
        img = Image.open(img_path)
        loc = loc[loc[:, 2] > 0.95]
        plt.figure()
        plt.imshow(img)
        plt.scatter(loc[:, 0], loc[:, 1], c='r', s=100 * (loc[:, 2] ** 15))
    plt.show()
    print(prediction)
