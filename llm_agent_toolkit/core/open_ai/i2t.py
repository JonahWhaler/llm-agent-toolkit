from llm_agent_toolkit._core import Core
from llm_agent_toolkit._util import (
    OpenAIRole, OpenAIMessage, OpenAIFunction, ContextMessage, ChatCompletionConfig,
)
import os
import openai
import base64


class I2T_OAI_Core(Core):
    """
    **Notes:**
    - Supported image format: png, jpeg, gif, webp
    """
    SUPPORTED_IMAGE_FORMATS = ["png", "jpeg", "gif", "webp"]

    def __init__(
            self, system_prompt: str, model_name: str, config: ChatCompletionConfig = ChatCompletionConfig(),
            tools: list | None = None
    ):
        super().__init__(
            system_prompt, model_name, config, tools
        )

    async def run_async(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        msgs = [{"role": OpenAIRole.SYSTEM.value, "content": self.system_prompt}]
        if context is not None:
            for ctx in context:
                if isinstance(ctx, OpenAIMessage):
                    msgs.append(ctx.__dict__())
                else:
                    msgs.append(ctx)
        filepath = kwargs.get("filepath", None)
        if filepath is not None:
            img_url = self.get_image_url(filepath)
            msgs.append(
                {"role": OpenAIRole.USER.value, "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        }
                    }
                ]}
            )
        msgs.append({"role": OpenAIRole.USER.value, "content": query})
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                n=self.config.n,
                functions=self.tools
            )
            choices = response.choices
            output = []
            for choice in choices:
                oam = OpenAIMessage(
                    role=OpenAIRole(choice.message.role),
                    content=choice.message.content,
                )
                if choice.message.tool_calls:
                    oam.functions = []
                    for tool_call in choice.message.tool_calls:
                        oam.functions.append(
                            OpenAIFunction(
                                call_id=tool_call.id,
                                function_name=tool_call.function.name,
                                args=tool_call.function.arguments
                            )
                        )
                output.append(oam)
            return output
        except Exception as e:
            print(f"run: {e}")
            raise

    @staticmethod
    def get_image_url(filepath: str):
        ext = filepath.split(".")[-1].lower()
        ext = "jpeg" if ext == "jpg" else ext
        if ext not in I2T_OAI_Core.SUPPORTED_IMAGE_FORMATS:
            raise Exception(f"Unsupported image type: {ext}")
        prefix = "data:image/{};base64".format(ext)
        try:
            with open(filepath, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
                return f"{prefix},{encoded_image}"
        except Exception as e:
            print(f"get_image_url: {e}")
            raise
