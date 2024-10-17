from transformers import MobileNetV2Config
from surya.settings import settings


class FontSizeModelConfig(MobileNetV2Config):
    model_type = "font_size_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = kwargs.get("num_labels", settings.SCALE_NUM_BUCKETS)
        self.problem_type = "single_label_classification"
        self.return_dict = False
