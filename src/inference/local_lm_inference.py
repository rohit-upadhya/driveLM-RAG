from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


class LocalInference:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._load_model()
        self._load_processor()

    def _load_model(
        self,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        pass

    def _load_processor(
        self,
    ):
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

    def generate_response(
        self,
        message: list[dict],
    ):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
