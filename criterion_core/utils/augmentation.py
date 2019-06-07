import imgaug.augmenters as ia

import criterion_core.utils.vue_schema_utils as schema
from criterion_core.utils import image_proc


def augmentation_schema(enable=True, vertical_translation=0, horizontal_translation=0, scaling=0, rotation=5, shear=0,
                        flipl=False, flipr=False, color_correct=False):
    return [
        schema.label("Data Augmentation"),
        schema.checkbox("Enable data augmentation?", "data_augmentation.enable", enable, False),
        schema.number_input("Vertical translation range in percent", "data_augmentation.vertical_translation_range",
                            default=vertical_translation, required=True, min=0, max=100),
        schema.number_input("Horizontal translation range in percent", "data_augmentation.horizontal_translation_range",
                            default=horizontal_translation, required=True, min=0, max=100),
        schema.number_input("Scaling range in percent", "data_augmentation.scaling_range",
                            default=scaling, required=True, min=0, max=100),
        schema.number_input("Rotation range in degrees", "data_augmentation.rotation_angle",
                            default=rotation, required=True, min=0, max=180),
        schema.number_input("Vertical translation range in percent", "data_augmentation.vertical_translation_range",
                            default=shear, required=True, min=0, max=1, step=0.1),
        schema.checkbox("Can images be flipped horizontally?", "data_augmentation.flip_horizontal", flipl, True),
        schema.checkbox("Can images be flipped vertically?", "data_augmentation.flip_vertical", flipr, True),
        schema.checkbox("Color correction", "data_augmentation.color_correct", color_correct, True),
    ]


def get_augmentation_pipeline(settings, target_shape, rois):
    augmentation = settings

    def rat(images, random_state, parents, hooks):
        aug = image_proc.random_affine_transform(target_shape, augmentation)
        images = image_proc.apply_transforms(images, aug, rois, target_shape)
        return images

    return ia.Sequential(
        ia.Lambda(
            rat
        )
    )
