from surya.settings import settings
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from surya.model.scale.model import FontSizeModel
from surya.input.processing import convert_if_not_rgb

FONT_SIZES = np.exp(np.linspace(np.log(settings.SCALE_MIN_FONT_SIZE), np.log(settings.SCALE_MAX_FONT_SIZE), settings.SCALE_NUM_BUCKETS))
TRANSFORMS = transforms.Compose(
    [transforms.Resize((settings.SCALE_MODEL_IMAGE_SIZE["width"], settings.SCALE_MODEL_IMAGE_SIZE["height"])), transforms.ToTensor()]
)


def get_font_size_from_bucket(bucket_index):
    return FONT_SIZES[bucket_index]


def get_bucket_from_font_size(font_size):
    closest_bucket = min(range(settings.NUM_BUCKETS), key=lambda i: abs(FONT_SIZES[i] - font_size))
    return closest_bucket


def get_scale_batch(images: list[Image.Image], scale_model: FontSizeModel):
    assert all(isinstance(image, Image.Image) for image in images)
    images = convert_if_not_rgb(images)

    with torch.inference_mode():
        pixel_values = torch.stack([TRANSFORMS(image) for image in images]).half()
        pixel_values = pixel_values.to(settings.TORCH_DEVICE_MODEL)
        logits = scale_model(pixel_values=pixel_values)[0]
        bucket_indices = logits.argmax(dim=1).tolist()
        font_sizes = [get_font_size_from_bucket(bucket_index) for bucket_index in bucket_indices]

    return font_sizes
