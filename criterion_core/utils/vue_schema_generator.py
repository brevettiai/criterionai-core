import json

import criterion_core.utils.vue_schema_utils as schema
from criterion_core.utils.augmentation import augmentation_schema
from criterion_core.utils.data_generator import COLOR_MODES


def data_schema(input_shape=(224, 224), class_mapping={"gode": "A", "chips": ["B", "C"]}):
    return [
        schema.label("Data"),
        schema.number_input("Network input width", "input_width",
                            default=input_shape[0], required=True, min=96, max=4096),
        schema.number_input("Network input height", "input_height",
                            default=input_shape[1], required=True, min=96, max=4096),
        schema.select("Color mode", "color_mode", required=True, default="greyscale",
                      values=list(COLOR_MODES.keys())),
        schema.text_input("Image rois. E.g. [[[100, 100], [400, 500]], [[500, 100], [800, 500]]]", "rois",
                          required=False, default="[]"),
        schema.text_input("Output classes as list", "classes", required=False),
        schema.text_area("Json class mapping", "class_mapping",
                         default=json.dumps(class_mapping, indent=2, sort_keys=True),
                         required=True,
                         placeholder="Map classes to folder names",
                         max=5000, rows=4, hint="Max 5000 characters")
    ]


def training_schema(epochs=10, batch_size=32, loss_function="weighted_binary_crossentropy",
                    class_weight={"chips": 10}):
    return [
        schema.label("Training"),
        schema.number_input("Epochs", "epochs",
                            default=epochs, required=True, min=1, max=50),
        schema.number_input("Batch size", "batch_size",
                            default=batch_size, required=True, min=1, max=64),
        schema.select("Loss function", "loss", default=loss_function, required=True,
                      values=[
                          "weighted_binary_crossentropy",
                          "binary_crossentropy",
                      ]),
        schema.text_area("Json class weights", "class_weight",
                         default=json.dumps(class_weight, indent=2, sort_keys=True),
                         required=True,
                         placeholder="Weight some classes more than others",
                         max=5000, rows=4, hint="Max 5000 characters")
    ]


def validation_schema(split_strategy="by_class", split_percentage=20):
    return [
        schema.label("Validation"),
        schema.select("Splitting strategy", "validation.split_strategy", default=split_strategy, required=True,
                      values=[
                          "by_class",
                      ]),
        schema.number_input("Split percentage", "validation.split_percentage",
                            default=split_percentage, required=True, min=0, max=100),
    ]


def export_schema(img_format="bmp"):
    return [
        schema.label("Export"),
        schema.select("Input image format in production", "img_format", default=img_format, required=True,
                      values=[
                          "bmp",
                          "jpeg",
                          "png",
                      ]),
    ]


def get_default_schema():
    fields = data_schema() + training_schema() + validation_schema() + augmentation_schema() + export_schema()
    return dict(fields=fields)


def generate_model_parameter_schema(path="model/settings-schema.json", schema=get_default_schema()):
    with open(path, "w") as fp:
        json.dump(schema, fp, indent=2, sort_keys=True)

    required_manifest = 'include %s' % path
    with open("MANIFEST.in", "a+") as fp:
        fp.seek(0)
        for line in fp:
            if required_manifest == line:
                return
        else:
            fp.write("\n" + required_manifest)
