from src.inference.open_ai_inference import OpenAICaller
from src.inference.local_lm_inference import LocalInference
from src.util.data_types import ModelType


class Inference:
    def __init__(
        self,
        model_type: ModelType = ModelType.OPEN_AI,
    ):
        self.model_type = model_type
        self.model = self._load_inference_obj()

    def _load_inference_obj(
        self,
    ):
        if self.model_type == ModelType.OPEN_AI:
            return OpenAICaller()
        if self.model_type == ModelType.CLIP:
            return LocalInference()

    def inference(
        self,
        input_prompt: list[dict],
    ):
        response = self.model.generate_response(input_prompt)
        print(f"LM response : {response}")
        return response
