import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


class LocalInference:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._load_quant_config(
            compute_dtype=compute_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        self.model_name_or_path = model_name_or_path
        self._load_model()
        self._load_processor()

    def _load_model(
        self,
    ):
        model_configs = {
            "torch_dtype": (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
            "device_map": "auto",
        }
        if self.quant_cfg is not None:
            model_configs["quantization_config"] = self.quant_cfg
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            **model_configs,
        )
        pass

    def _load_quant_config(
        self,
        compute_dtype: torch.dtype,
        load_in_4bit: bool,
        load_in_8bit: bool,
    ):
        self.quant_cfg = None

        if load_in_4bit or load_in_8bit:
            self.quant_cfg = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

    def _load_processor(
        self,
    ):
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

    def generate_response(
        self,
        messages: list[dict],
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
