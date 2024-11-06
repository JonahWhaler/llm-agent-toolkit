import os
import openai
from dataclasses import dataclass


class EmbeddingModel:
    def __init__(
            self, embedding_model: str = "text-embedding-3-small",
    ):
        self.__embedding_model = embedding_model

    def text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.__embedding_model)
            return response.data[0].embedding
        except Exception as e:
            print(f"text_to_embedding: {e}")
            raise


@dataclass
class ChatCompletionConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    n: int = 1


class ChatCompletionModel:
    def __init__(
            self, gpt_model_name: str = "gpt-4o-mini", config: ChatCompletionConfig = ChatCompletionConfig(),
    ):
        self.gpt_model_name = gpt_model_name
        self.config = config

    def generate(self, system_prompt: str, query: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=self.gpt_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                n=self.config.n
            )
            choices = response.choices
            output = [choice.message.content.strip() for choice in choices]
            return output
        except Exception as e:
            print(f"generate: {e}")
            raise
