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


class AffineTransformation(ia.Augmenter):
    """
    Wrapper for original affine transformation code to imgaug
    """
    def __init__(self, settings, target_shape, rois,
                 name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.target_shape = target_shape
        self.settings = settings
        self.rois = rois

    def _augment_images(self, images, random_state, parents, hooks):
        aug = image_proc.random_affine_transform(self.target_shape, self.settings)
        images = image_proc.apply_transforms(images, aug, self.rois, self.target_shape)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError

    def _augment_polygons(self, polygons_on_images, random_state, parents, hooks):
        raise NotImplementedError

    def get_parameters(self):
        return []

def get_augmentation_pipeline(settings, target_shape, rois):

    if settings is not None and settings.enable:
        return ia.Sequential(
            AffineTransformation(settings, target_shape, rois)
        )
    else:
        return ia.Sequential()
