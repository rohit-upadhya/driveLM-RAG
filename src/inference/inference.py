from src.inference.open_ai_inference import OpenAICaller
from src.util.data_types import ModelType


class Inference:
    def __init__(
        self,
        input_prompt: list[dict],
        model_type: ModelType = ModelType.OPEN_AI,
    ):
        self.model_type = model_type
        self.input_prompt = input_prompt

    def _get_open_ai_inference(
        self,
        input_prompt: list[dict],
    ):
        api_caller = OpenAICaller()
        return api_caller.generate_response(message=input_prompt)

    def _get_local_inference(
        self,
        input_prompt: list[dict],
    ):
        pass

    def inference(
        self,
    ):
        if self.model_type == ModelType.OPEN_AI:
            return self._get_open_ai_inference(input_prompt=self.input_prompt)
