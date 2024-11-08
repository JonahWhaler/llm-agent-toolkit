from llm_agent_toolkit._core import Core
from llm_agent_toolkit._util import (
    OpenAIRole, OpenAIMessage, ContextMessage, ImageGenerationConfig
)
import os
import openai
import base64


class T2I_OAI_Core(Core):
    def __init__(
            self, system_prompt: str, model_name: str, config: ImageGenerationConfig = ImageGenerationConfig(),
            tools: list | None = None
    ):
        super().__init__(
            system_prompt, model_name, config, tools
        )
        assert config.response_format == "b64_json"

    async def run_async(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        username: str = kwargs.get("user_name", "User")
        params = self.config.__dict__()
        params["model"] = self.model_name
        params["user"] = username
        params["prompt"] = query
        output = []
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            images_response = await client.images.generate(**params)
            for idx, image in enumerate(images_response.data):
                img_model = image.model_dump()
                img_b64 = img_model['b64_json']
                img_decoding = base64.b64decode(img_b64)
                export_path = f"./{username}_{idx}.png"
                with open(export_path, "wb") as f:
                    f.write(img_decoding)
                output.append(
                    OpenAIMessage(
                        role=OpenAIRole.ASSISTANT, content=export_path
                    )
                )

            return output
        except Exception as e:
            print(f"run_async: {e}")
            raise
