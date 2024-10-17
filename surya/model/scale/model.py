from transformers import MobileNetV2ForImageClassification
from surya.model.scale.config import FontSizeModelConfig
from surya.settings import settings


class FontSizeModel(MobileNetV2ForImageClassification):
    base_model_prefix = "model"
    config_class = FontSizeModelConfig


def load_model(checkpoint=settings.SCALE_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    config = FontSizeModelConfig()
    model = FontSizeModel.from_pretrained(checkpoint, torch_dtype=dtype, config=config)

    model = model.to(device)
    model = model.eval()

    print(f"Loaded scale model {checkpoint} on device {device} with dtype {dtype}")
    return model
