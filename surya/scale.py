from surya.settings import settings
import numpy as np
import torch
from PIL import Image
from surya.model.scale.model import FontSizeModel
from surya.input.processing import convert_if_not_rgb
from transformers import AutoImageProcessor

FONT_SIZES = np.exp(np.linspace(np.log(settings.SCALE_MIN_FONT_SIZE), np.log(settings.SCALE_MAX_FONT_SIZE), settings.SCALE_NUM_BUCKETS))


def get_font_size_from_bucket(bucket_index):
    return FONT_SIZES[bucket_index]


def get_bucket_from_font_size(font_size):
    closest_bucket = min(range(settings.NUM_BUCKETS), key=lambda i: abs(FONT_SIZES[i] - font_size))
    return closest_bucket


def get_scale_batch(images: list[Image.Image], scale_model: FontSizeModel, scale_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(settings.SCALE_MODEL_CHECKPOINT)):
    assert all(isinstance(image, Image.Image) for image in images)
    images = convert_if_not_rgb(images)

    with torch.inference_mode():
        pixel_values = torch.stack([scale_processor(image, return_tensors="pt")["pixel_values"].squeeze(0) for image in images]).half()
        pixel_values = pixel_values.to(settings.TORCH_DEVICE_MODEL)
        logits = scale_model(pixel_values=pixel_values)[0]
        bucket_indices = logits.argmax(dim=1).tolist()
        font_sizes = [get_font_size_from_bucket(bucket_index) for bucket_index in bucket_indices]
    font_sizes = [reverse_rescale_font_size(image.size[0], image.size[1], font_size) for image, font_size in zip(images, font_sizes)]
    return font_sizes


def reverse_rescale_font_size(
    image_width, image_height, new_font_size,
    new_width=settings.SCALE_MODEL_IMAGE_SIZE["width"],
    new_height=settings.SCALE_MODEL_IMAGE_SIZE["height"]
):
    scaling_factor_1 = new_width / image_width
    scaling_factor_2 = new_height / image_height
    average_scaling_factor = (scaling_factor_1 + scaling_factor_2) / 2
    original_font_size = new_font_size / average_scaling_factor
    return original_font_size
