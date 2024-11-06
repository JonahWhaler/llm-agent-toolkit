import json
from typing import Optional
from datetime import datetime
from .base import BaseTool, ToolOutput, ToolOutputItem, ToolError
from .util import ChatCompletionModel, ChatCompletionConfig

QUESTION_SUGGESTER_AGENT_PROMPT = """
# Follow-Up Question Suggester Agent

You are Follow-Up Suggester, an AI agent specialized in suggesting relevant follow-up questions that users can ask the AI based on the conversation context.

**Input Format:**
QUERY={user's original query}
RESPONSE={AI model's response}

**Primary Functions:**
- Parse conversation context
- Generate relevant follow-up questions users can ask the AI
- Suggest deeper exploration paths
- Identify related topics users might be interested in
- Surface valuable angles users might not have considered

**Analysis Framework:**
1. Context Analysis
   - Core topic identification
   - Response comprehensiveness
   - Potential areas of expansion
   - Related concepts worth exploring

2. Question Categories:
   a) Deep Dive
      - Request detailed explanations
      - Ask for specific examples
      - Example: "Can you explain more about [specific concept mentioned]?"

   b) Practical Application
      - Implementation details
      - Real-world usage
      - Example: "Could you provide a step-by-step guide for implementing [mentioned solution]?"

   c) Alternatives & Comparisons
      - Different approaches
      - Trade-offs
      - Example: "What are some alternative approaches to [mentioned solution]?"

   d) Extensions
      - Related topics
      - Advanced concepts
      - Example: "How does this work with [related technology]?"

**Required Output Format:**
[
    {
        "follow_up": "Suggested question user can ask the AI",
        "purpose": "Why this question would be valuable to ask",
        "expected_insight": "What user will learn from AI's response"
    }
]

**Example:**
Input:
QUERY=How do I use async/await in JavaScript?
RESPONSE=Async/await is a way to handle asynchronous operations in JavaScript. The 'async' keyword declares an asynchronous function, while 'await' pauses execution until a promise resolves. Here's a basic example... [example code provided]

Output:
[
    {
        "follow_up": "Can you show me some common error handling patterns with async/await?",
        "purpose": "Learn about robust error handling in asynchronous code",
        "expected_insight": "Best practices for managing errors in async/await operations"
    },
    {
        "follow_up": "How does async/await compare to using regular Promises?",
        "purpose": "Understand trade-offs between different async patterns",
        "expected_insight": "Comparative analysis of async/await vs Promises, with pros and cons"
    },
    {
        "follow_up": "Can you demonstrate async/await usage in a real-world API call scenario?",
        "purpose": "See practical application in common use case",
        "expected_insight": "Concrete example of implementing async/await with API calls"
    }
]

**Guidelines:**
- Generate questions users can directly ask the AI
- Ensure questions build upon the original context
- Focus on questions that will yield valuable AI responses
- Progress from basic to more advanced concepts
- Consider different learning angles

**Question Types to Generate:**
1. First question: Deeper understanding of core concept
2. Second question: Practical application or comparison
3. Third question: Advanced usage or extension

**Response Rules:**
1. Return valid JSON array of suggested questions
2. Each question should:
   - Be something the AI can effectively answer
   - Build on the existing context
   - Provide new valuable information
3. Return "NA" for:
   - Simple commands
   - Complete technical instructions
   - Queries that don't warrant follow-ups

Remember:
- Questions should be from user's perspective TO the AI
- Each question should deliver unique value
- Questions should help users get the most out of the AI's capabilities
"""


class QuestionSuggesterAgent(BaseTool):
    def __init__(
            self, priority: int = 1, next_func: Optional[str] = None,
            ccm: ChatCompletionModel = ChatCompletionModel(),
            debug: bool = False
    ):
        super().__init__(
            tool_name="QuestionSuggesterAgent",
            description=QUESTION_SUGGESTER_AGENT_PROMPT,
            priority=priority,
            next_func=next_func
        )
        self.__ccm = ccm
        self.__debug = debug

    def __call__(self, params: str) -> ToolOutput:
        # Validate parameters
        if not self.validate(params):
            return ToolOutput(
                tool_name=self.name,
                error=ToolError(
                    type="validation_error",
                    message="Invalid parameters for QueryExpanderAgent"
                )
            )
        # Load parameters
        params = json.loads(params)
        p_query = params.get("query", None)
        p_response = params.get("response", None)
        augmented_query = f"QUERY={p_query}\nRESPONSE={p_response}"
        output = ToolOutput(tool_name=self.name, result=[])
        try:
            llm_results = self.__ccm.generate(self.description, augmented_query)
            for i, llm_result in enumerate(llm_results):
                json_questions = json.loads(llm_result)
                for question in json_questions:
                    _q = {"question": question.get("follow_up", None)}
                    if self.__debug:
                        _q["purpose"] = question.get("purpose", None)
                        _q["expected_insight"] = question.get("expected_insight", None)
                    item = ToolOutputItem(
                        identifier=f"question_suggester_result_{i}",
                        value=json.dumps(_q),
                        timestamp=datetime.now().isoformat(),
                        is_answer=True
                    )
                    output.result.append(item)
            return output
        except json.JSONDecodeError as jde:
            return ToolOutput(
                tool_name=self.name,
                error=ToolError(
                    type="json_decode_error",
                    message=str(jde)
                )
            )
        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                error=ToolError(
                    type="error",
                    message=str(e)
                )
            )

    def validate(self, params: str):
        params = json.loads(params)
        p_query = params.get("query", None)
        p_response = params.get("response", None)
        if p_query is None or p_response is None:
            return False
        p_query = p_query.strip()
        p_response = p_response.strip()
        conditions = [len(p_query) > 0, len(p_response) > 0]
        if not all(conditions):
            return False
        return True
