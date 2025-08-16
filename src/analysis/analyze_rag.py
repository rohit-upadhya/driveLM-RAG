from html import escape
from PIL import Image
from IPython.display import HTML, display

class AnalyzeRag:
    def __init__(
        self,
        rag_data: dict,
    ) -> None:
        self.rag_data = rag_data
        self.relevant_data = self._pick_relevant_info()
        pass

    def _pick_relevant_info(
        self,
    ):
        relevant_data = []
        for _, v in self.rag_data.items():
            scene_description = v["scene_description"]
            for _, key_frame in v["key_frames"].items():
                qa = key_frame["QA"]
                for question_type, questions in qa.items():
                    for question in questions:
                        images = []
                        for sample in question["retrieval"]:
                            images.append({
                                "image": sample["image_pil"],
                                "objects": sample["objects"],
                                "speed": sample["ego"]["speed_mps"]
                            })
                        relevant_data.append({
                            "scene_description": scene_description,
                            "question": question["Q"],
                            "answer": question["A"],
                            "cosine": question["cosine"],
                            "generated_answer": question["generated_answer"],
                            "image_dict": images

                        })
    
        return relevant_data
    
    def _display_question_answer(self,question, answer, generated_answer, cosine):
        question_safe = escape(question)
        answer_safe = escape(answer)

        html = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.5; margin-bottom: 20px;">
            <div style="font-size:18px; font-weight:bold; color:#007acc; margin-bottom:8px;">
                Question:
            </div>
            <div style="font-size:16px; color:#1a1a1a; margin-bottom:15px; white-space: pre-wrap;">
                {question_safe}
            </div>
            <div style="font-size:18px; font-weight:bold; color:#cc0000; margin-bottom:8px;">
                Answer:
            </div>
            <div style="font-size:16px; color:#333333; white-space: pre-wrap;">
                {answer_safe}
            </div>
            <div style="font-size:18px; font-weight:bold; color:#cc0000; margin-bottom:8px;">
                Generated Answer:
            </div>
            <div style="font-size:16px; color:#333333; white-space: pre-wrap;">
                {generated_answer}
            </div>
            <div style="font-size:18px; font-weight:bold; color:#cc0000; margin-bottom:8px;">
                Cosine Similarity:
            </div>
            <div style="font-size:16px; color:#333333; white-space: pre-wrap;">
                {cosine}
            </div>
            <div style="font-size:18px; font-weight:bold; color:#cc0000; margin-bottom:8px;">
                Retrieved Images with Metadata:
            </div>
        </div>
        """
        display(HTML(html))

    def _display_image_information(
        self,
        image_list: list,
    ):
        for item in image_list:
            print(f"Speed : {item['speed']}")
            print(f'Objects : {item["objects"]}')
            image = item["image"]
            display(image)
    def display_information(
        self,
    ):
        for item in self.relevant_data:
            self._display_question_answer(question=item["question"], answer=item["answer"], generated_answer=item["generated_answer"], cosine=item["cosine"])

            self._display_image_information(image_list=item["image_dict"])
    
