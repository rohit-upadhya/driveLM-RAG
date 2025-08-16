import json
import os
import numpy as np
import base64

from PIL import Image
from io import BytesIO

from src.encoders.encoder import EncodeImageText
from src.encoders.faiss import FaissDB
from src.inference.inference import Inference
from src.util.prompter import Prompter
from src.util.load_file import FileLoader
from src.util.data_types import ModelType


class LMRag:
    def __init__(
        self,
        drive_lm_data_file: str = "resources/data/drivelm/train_sample.json",
        nuscenes_file: str = "resources/output/nusciences_processed.json",
        prompt_template_file: str = "resources/prompt_template.yaml",
        local_inference: bool = False,
    ) -> None:
        self.file_loader = FileLoader()
        self.drive_lm_data = self.file_loader.load_file(drive_lm_data_file)
        self.nuscenes_data = self.file_loader.load_file(nuscenes_file)
        self.ids = []
        self.encoder = EncodeImageText()
        self.faiss = FaissDB()
        self.prompt_template_file = prompt_template_file
        self.local_inference = local_inference
        self._load_inference()

    def _load_inference(
        self,
    ):
        model_type = ModelType.OPEN_AI
        if self.local_inference:
            model_type = ModelType.CLIP
        self.inference_obj = Inference(model_type=model_type)
        pass

    def _inference(
        self,
        message: list[dict],
    ):
        inference = self.inference_obj.inference(input_prompt=message)
        return inference[0] if isinstance(inference, list) else inference

    def _build_index(
        self,
        all_encodings: np.ndarray,
        ids: list,
    ):
        self.faiss.build_index(all_encodings, ids)
        return

    def _build_prompt(
        self,
        query: str,
        images_dict: list,
    ):
        query_items = []
        for idx, item in enumerate(images_dict):
            text = f"Metadata for image : {idx+1}"
            if "speed" in item:
                text = f'speed : {item["speed"]}\n\n'
            if "objects" in item:
                text = f'{text}Objects present : {item["objects"]}\n'
            query_items.append(
                {
                    "type": "text",
                    "item": text,
                }
            )
            query_items.append(
                {
                    "type": "image",
                    "item": item["image_base_64"],
                }
            )
        query_items.append(
            {
                "type": "text",
                "item": f"Final query based on previous images and metadata : \n\n{query}.",
            }
        )
        prompt_template = self.file_loader.load_file(self.prompt_template_file).get(
            "rag", {}
        )
        prompter = Prompter(
            prompt_template=prompt_template,
            query_items=query_items,
            local_inference=self.local_inference,
        )
        return prompter.build_chat_prompt()

    def _save_json_files(
        self,
        data: list | dict,
        file_name: str,
    ) -> None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w+") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def _encode_query(
        self,
        query: str,
    ):
        encoded_query = self.encoder.encode_text(texts=[query])
        return encoded_query

    def _process_image(
        self,
        image_path: str,
    ):
        img = Image.open(image_path).convert("RGB")

        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img, img_base64

    def _retreival(
        self,
        query,
    ) -> list:
        encoded_query = self._encode_query(query=query)
        print(encoded_query.shape)
        relevant_ids = self.faiss.perform_search(query=encoded_query)
        relevant_samples = []
        for id in relevant_ids:
            relevant_sample = self.nuscenes_data[id]
            relevant_sample["image_pil"], relevant_sample["image_base_64"] = (
                self._process_image(image_path=relevant_sample["image"]["rel_path"])
            )
            relevant_samples.append(relevant_sample)
        return relevant_samples

    def _encode_images(
        self,
        images: list,
    ):
        image_encodings = self.encoder.encode_image(images=images)
        print("encoded all images.")
        return image_encodings

    def perform_rag(
        self,
    ):
        images = []

        for idx, item in enumerate(self.nuscenes_data):
            images.append(item["image"]["rel_path"])
            self.ids.append(idx)

        encoded_images = self._encode_images(images=images)
        self._build_index(all_encodings=encoded_images, ids=self.ids)

        for k_1, v_1 in self.drive_lm_data.items():
            scene_description = v_1["scene_description"]
            relevant_samples = self._retreival(query=scene_description)
            key_frames = v_1["key_frames"]
            for k_2, v_2 in key_frames.items():
                qa = v_2["QA"]
                for type, questions in qa.items():
                    for qi, q in enumerate(questions):
                        query_text = q.get("Q") if isinstance(q, dict) else str(q)
                        prompt = self._build_prompt(
                            query=query_text, images_dict=relevant_samples
                        )
                        lm_answer = self._inference(message=prompt)
                        print(lm_answer)
                        if isinstance(q, dict):
                            questions[qi]["generated_answer"] = lm_answer
                        else:
                            questions[qi] = {
                                "Q": query_text,
                                "generated_answer": lm_answer,
                            }
        self._save_json_files(
            data=self.drive_lm_data,
            file_name="resources/output/drive_lm_with_lm_outputs.json",
        )


if __name__ == "__main__":
    lm_rag = LMRag()
    lm_rag.perform_rag()
