import os
import argparse
import logging
from tensorflow_serving.config import model_server_config_pb2

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='model_dir', help='Model directory', nargs='?', default='models')
    args, unparsed = parser.parse_known_args()
    return args, unparsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args, _ = parse_args()

    model_server_config = model_server_config_pb2.ModelServerConfig()
    base, folders, files = next(os.walk(args.model_dir))
    for folder in folders:
        model_config = model_server_config_pb2.ModelConfig()
        model_config.model_platform = "tensorflow"
        model_config.name = folder
        model_config.base_path = os.path.join(os.path.abspath(base), folder)

        ## Logging config not working in tf serving version 1.13
        #model_config.logging_config.sampling_config.sampling_rate = 0.1
        #model_config.logging_config.log_collector_config.type = "file"
        #model_config.logging_config.log_collector_config.filename_prefix = "log_"

        model_server_config.model_config_list.config.extend([model_config])

    log.info(str(model_server_config))

    with open("models.config", "w") as fp:
        fp.write(str(model_server_config))