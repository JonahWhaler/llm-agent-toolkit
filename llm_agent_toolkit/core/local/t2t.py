# import os
import warnings
import logging
import ollama
from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)

# from ..._tool import Tool


logger = logging.getLogger(__name__)


class T2T_OLM_Core(Core):
    def __init__(self, system_prompt: str, config: ChatCompletionConfig):
        assert isinstance(config, ChatCompletionConfig)
        super().__init__(system_prompt, config, None)

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]
        if context is not None:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))
        number_of_primers = len(msgs)
        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 4096
        iteration = 0
        token_count = 0
        solved = False
        try:
            client = ollama.Client(host="http://localhost:11434")
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                logger.info(response)
                llm_generated_content = response["message"]["content"]
                msgs.append(
                    MessageBlock(
                        role=CreatorRole.ASSISTANT.value, content=llm_generated_content
                    )
                )
                iteration += 1
                break

            if not solved:
                if iteration == self.config.max_iteration:
                    warnings.warn(
                        f"Maximum iteration reached. {iteration}/{self.config.max_iteration}"
                    )
                elif token_count >= max_tokens:
                    warnings.warn(
                        f"Maximum token count reached. {token_count}/{max_tokens}"
                    )
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            logger.error(f"run: {e}")
            raise

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]
        if context is not None:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))
        number_of_primers = len(msgs)
        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 4096
        iteration = 0
        token_count = 0
        solved = False
        try:
            client = ollama.AsyncClient(host="http://localhost:11434")
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                logger.info(response)
                llm_generated_content = response["message"]["content"]
                msgs.append(
                    MessageBlock(
                        role=CreatorRole.ASSISTANT.value, content=llm_generated_content
                    )
                )
                iteration += 1
                break

            if not solved:
                if iteration == self.config.max_iteration:
                    warnings.warn(
                        f"Maximum iteration reached. {iteration}/{self.config.max_iteration}"
                    )
                elif token_count >= max_tokens:
                    warnings.warn(
                        f"Maximum token count reached. {token_count}/{max_tokens}"
                    )
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            logger.error(f"run: {e}")
            raise
