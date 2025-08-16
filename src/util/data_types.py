from enum import Enum


class ErrorTypes(Enum):
    pass


class ModelType(Enum):
    OPEN_AI = "open_ai"
    CLIP = "clip"
    QWEN = "qwen"
