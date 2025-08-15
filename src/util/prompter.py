from typing import Any


class Prompter:
    def __init__(
        self,
        prompt_template: dict[Any, Any],
        query: str,
        images: list | None = None,
    ):
        self.prompt_template = prompt_template
        self.query = query
        self.images = images
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
        if self.query:
            content.append(
                {
                    "type": "input_text",
                    "text": self.query,
                }
            )
        else:
            raise ValueError("No text provided. Please provide a query and try again.")
        if self.images:

            for image in self.images:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image}",
                    },
                )

        final_prompt.append({"role": "user", "content": content})
        return final_prompt
