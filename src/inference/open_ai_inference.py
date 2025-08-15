import os
import base64

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()


class OpenAICaller:

    def __init__(
        self,
    ):
        self.client = self.initialize_open_ai_client()

    def initialize_open_ai_client(
        self,
    ):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client
        except Exception as e:
            raise ValueError(
                f"Following error occurred while trying to initialize Open-AI client: {e}."
            )

    def generate_response(
        self,
        message: list[dict] | None = None,
    ):
        model = "gpt-4.1"
        try:
            response = self.client.responses.create(
                model=model,
                input=message,
            )
            return response.output_text

        except (RateLimitError, APIError) as e:
            raise ValueError(f"error : {e} occured while trying to get response")

        except Exception as e:
            raise ValueError(f"error : {e} occured while trying to get response")


if __name__ == "__main__":
    image_path = "resources/data/drivelm/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915244537570.jpg"

    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

    input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "what is going on here?",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image}",
                },
            ],
        }
    ]
    api_caller = OpenAICaller()

    print(api_caller.generate_response(message=input))
