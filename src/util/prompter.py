from typing import Any


class Prompter:
    def __init__(
        self,
        prompt_template: dict[Any, Any],
        query_items: list[Any],
        local_inference: bool = False,
    ):
        self.prompt_template = prompt_template
        self.query_items = query_items
        self.local_inference = local_inference
        if not self.prompt_template:
            raise ValueError(
                "Prompt template is not provided. Please provide a prompt template for further processing."
            )
        pass

    def build_chat_prompt(
        self,
    ) -> list[dict[str, str]]:
        final_prompt = []
        print(self.prompt_template)
        if "system" in self.prompt_template:
            content = self.prompt_template.get("system", "")
            final_prompt.append({"role": "system", "content": content})
        content = []
        if self.query_items:
            for item in self.query_items:
                type = item["type"]
                item_individual = item["item"]
                content.append(
                    {
                        "type": "input_text" if type == "text" else "input_image",
                        "text": (
                            item_individual
                            if type == "text"
                            else f"data:image/jpeg;base64,{item_individual}"
                        ),
                    }
                )
        else:
            raise ValueError("No text provided. Please provide a query and try again.")

        final_prompt.append({"role": "user", "content": content})
        return self.convert_to_local(final_prompt)

    def convert_to_local(
        self,
        messages: list[dict],
    ) -> list[dict]:
        if not self.local_inference:
            return messages

        out = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not isinstance(content, list):
                out.append({"role": role, "content": content})
                continue

            new_content = []
            for part in content:
                ptype = part.get("type")
                if ptype == "input_text":
                    new_content.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "input_image":
                    data = part.get("text") or part.get("image")
                    new_content.append({"type": "image", "image": data})
                elif ptype == "input_video":
                    data = part.get("text") or part.get("video")
                    new_content.append({"type": "video", "video": data})
                else:
                    new_content.append(part)

            out.append({"role": role, "content": new_content})
        return out
