from llm_agent_toolkit._core import Core
from llm_agent_toolkit._util import (
    OpenAIRole, OpenAIMessage, OpenAIFunction, ContextMessage, ChatCompletionConfig,
)
import os
import openai


class T2T_OAI_Core(Core):
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
                    content=choice.message.content.strip(),
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

    def run(self, query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        msgs = [{"role": OpenAIRole.SYSTEM.value, "content": self.system_prompt}]
        if context is not None:
            for ctx in context:
                if isinstance(ctx, OpenAIMessage):
                    msgs.append(ctx.__dict__())
                else:
                    msgs.append(ctx)
        msgs.append({"role": OpenAIRole.USER.value, "content": query})
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
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
                    content=choice.message.content.strip(),
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
