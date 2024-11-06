import json
import os
from dotenv import load_dotenv

from llm_agent_toolkit import (
    ChatCompletionConfig,
    ChatCompletionModel,
    EmbeddingModel,
    QueryExpanderAgent,
    QuestionSuggesterAgent
)


def run_chat_completion_model():
    llm = ChatCompletionModel(
        gpt_model_name="gpt-4o-mini",
        config=ChatCompletionConfig(max_tokens=256, temperature=0.7, n=1)
    )
    system_prompt = "You are a helpful assistant."
    user_prompt = "What can you do?"
    llm_results = llm.generate(system_prompt, user_prompt)
    for llm_result in llm_results:
        print(llm_result)


def run_embedding_model():
    encoder = EmbeddingModel()
    user_prompt = "You are awesome!"
    embedding = encoder.text_to_embedding(user_prompt)
    print(embedding)


def run_query_expander_agent():
    llm = ChatCompletionModel(
        gpt_model_name="gpt-4o-mini",
        config=ChatCompletionConfig(max_tokens=256, temperature=0.7, n=1)
    )
    agent = QueryExpanderAgent(ccm=llm)
    user_prompt = "Why would anyone consider legalizing abortions as an act to reclaim reproductive rights?"
    agent_response = agent(json.dumps({"query": user_prompt}))
    if agent_response.error:
        print(agent_response.error)
    else:
        for result in agent_response.result:
            print(result)


def run_question_suggester_agent():
    llm = ChatCompletionModel(
        gpt_model_name="gpt-4o-mini",
        config=ChatCompletionConfig(max_tokens=256, temperature=0.7, n=1)
    )
    agent = QuestionSuggesterAgent(
        ccm=llm, debug=True
    )
    user_prompt = "Why is the earth not flat?"
    sim_response = """
    The Earth is not flat; it's roughly spherical. Here's why:
    * **Photographs from Space:** Images taken from satellites and spacecraft clearly show the Earth's curvature.
    * **Ship Disappearing Over the Horizon:** As a ship sails away, its hull disappears first, then its mast, indicating a curved surface.
    * **Different Time Zones:** The Earth rotates, causing different parts to face the sun at different times, resulting in day and night. This wouldn't be possible on a flat Earth.
    * **Lunar Eclipse:** The Earth's shadow on the Moon during a lunar eclipse is always round, proving the Earth's spherical shape.
    * **Circumnavigation:** Explorers like Magellan circumnavigated the globe, proving it's not flat.
    
    These are just a few of the many pieces of evidence that confirm the Earth's spherical shape.
    """
    agent_response = agent(json.dumps(
        {"query": user_prompt, "response": sim_response}
    ))
    if agent_response.error:
        print(agent_response.error)
    else:
        for result in agent_response.result:
            print(result)


if __name__ == "__main__":
    load_dotenv()
    # run_chat_completion_model()
    # run_embedding_model()
    # run_query_expander_agent()
    run_question_suggester_agent()
