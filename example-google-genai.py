import os
import json
from typing import Type, TypeVar
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def list_models():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.list()
    for m in response.page:
        print(f">> {m}\n")


def text_generation(prompt: str, output_format: str):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt + f"\n{output_format}")],
        )
    ]
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=messages,  # type: ignore
        config=types.GenerateContentConfig(
            system_instruction="You are faithful AI chatbot.", temperature=0.7
        ),
    )
    if response.text:
        try:
            character_object = json.loads(response.text)
            for k, v in character_object.items():
                print(f"{k} : {v}")
        except Exception as e:
            print(f"Exception: {str(e)}")
            print(response.text)


class ImgDescription(BaseModel):
    title: str
    vibe: str
    short_summary: str
    long_description: str
    keywords: list[str]


def image_interpretation(prompt: str, filepath: str, response_format: Type[T]):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with open(filepath, "rb") as reader:
        image_url = reader.read()

    messages = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_url, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
        )
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,  # type: ignore
        config=types.GenerateContentConfig(
            system_instruction="You are a faithful AI chatbot.",
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=response_format,
        ),
    )
    # print(response.text)
    description: ImgDescription = response.parsed  # type: ignore
    jobj = description.model_dump()
    for k, v in jobj.items():
        print(f"{k} : {v}")


def generate_random_number(lower_limit: int, upper_limit: int) -> int:
    """Generate Random Number.

    Args:
        lower_limit (int): The smallest value, inclusive.
        upper_limit (int): The largest value, inclusive.

    Returns:
        output (int): Random Number.
    """
    import random as rn

    print("Roll", lower_limit, upper_limit)
    return rn.randint(lower_limit, upper_limit)


def tool_calling(prompt: str):
    # from llm_agent_toolkit.tool import LazyTool

    # dice = LazyTool(generate_random_number, is_coroutine_function=False)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=messages,  # type: ignore
        config=types.GenerateContentConfig(
            system_instruction="You are faithful AI chatbot.",
            temperature=0.7,
            tools=[generate_random_number],
        ),
    )
    print(f"Response: {response.usage_metadata}")
    print(f"FunctionCalls: {response.function_calls}")

    if response.text:
        print(response.text)


def tool_calling_v2(prompt: str, output_format: Type[T]):
    # from llm_agent_toolkit.tool import LazyTool

    # dice = LazyTool(generate_random_number, is_coroutine_function=False)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="roll_dice",
                description="Roll n-dimention dice.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "lower_limit": types.Schema(
                            type=types.Type.NUMBER,
                            description="Lower limit (inclusive)",
                        ),
                        "upper_limit": types.Schema(
                            type=types.Type.NUMBER,
                            description="Upper limit (inclusive)",
                        ),
                    },
                    required=["lower_limit", "upper_limit"],
                ),
            )
        ]
    )
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=messages,  # type: ignore
        config=types.GenerateContentConfig(
            system_instruction="You are faithful AI chatbot.",
            temperature=0.7,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    # allowed_function_names=["roll_dice"],
                )
            ),
        ),
    )

    print(f"Response: {response.usage_metadata}")
    print(f"FunctionCalls: {response.function_calls}")

    if response.function_calls:
        function_call = response.function_calls[0]
        function_name = function_call.name
        print(f"Calling {function_name}...")
        args = function_call.args
        if args:
            result = generate_random_number(**args)
            messages.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"{json.dumps(args)} => {result}")
                    ],
                )
            )
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=messages,  # type: ignore
                config=types.GenerateContentConfig(
                    system_instruction="You are faithful AI chatbot.",
                    temperature=0.7,
                    # tools=[tool],
                    # tool_config=types.ToolConfig(
                    #     function_calling_config=types.FunctionCallingConfig(
                    #         mode=types.FunctionCallingConfigMode.ANY,
                    #         allowed_function_names=["roll_dice"],
                    #     )
                    # ),
                    response_mime_type="application/json",
                    response_schema=output_format,
                ),
            )
            print(response)


CHARACTER_STRUCT = """
Note:
* Strictly response in JSON structure defined below without tripple ticks.

JSON Output:
{
    \"name\": {{Name:str}},
    \"gender\": {{Gender:str}},
    \"courage\": {{Courage:int}}
    \"defense\": {{Defense:int}},
    \"agility\": {{Agility:int}},
    \"career\": {{Warior|Archer|Magician}},
    \"background_story\": {{BackgroundStory}}
}
"""

IMAGE_DESCRIPTION_STRUCT = """
Note:
* Strictly response in JSON structure defined below without tripple ticks.

JSON Output:
{
    \"title\": {{Title:str}},
    \"vibe\": {{Vibe:str}},
    \"short_summary\": {{ShortSummary:str}},
    \"long_description\": {{LongDescription:str}},
    \"keywords\": [{{Keyword:str}}]
}
"""


class GeneralOutput(BaseModel):
    result: str


if __name__ == "__main__":
    load_dotenv()

    prompt_1 = "Create a random character."
    text_generation(prompt_1, CHARACTER_STRUCT)

    prompt_2 = "Describe this image."
    filepath = "./dev/image/wednesday-addams-00.jpg"
    image_interpretation(prompt_2, filepath, ImgDescription)

    prompt_3 = "Roll a dice!"
    tool_calling_v2(prompt_3, GeneralOutput)
