from typing import Any


class Prompter:
    def __init__(
        self,
        prompt_template: dict[Any, Any],
        query_items: list[Any],
    ):
        self.prompt_template = prompt_template
        self.query_items = query_items
        if not self.prompt_template:
            raise ValueError(
                "Prompt template is not provided. Please provide a prompt template for further processing."
            )
        pass

    def build_chat_prompt(
        self,
    ) -> list[dict[str, str]]:
        final_prompt = []
        if "system_prompt" in self.prompt_template:
            content = self.prompt_template.get("system_prompt", "")
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
        return final_prompt
