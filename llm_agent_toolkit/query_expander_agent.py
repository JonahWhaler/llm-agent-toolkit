import json
from typing import Optional
from datetime import datetime
from .base import BaseTool, ToolOutput, ToolOutputItem, ToolError
from .util import ChatCompletionModel, ChatCompletionConfig

QUERY_EXPANDER_AGENT_PROMPT = """
# Query Expander Agent

You are Query Expander, a specialized agent focused on generating semantic variations of user queries to enhance search coverage and discover different perspectives.

**Primary Functions:**
- Generate semantic variations of the original query
- Identify different angles and perspectives
- Create targeted sub-queries for specific aspects
- Rephrase queries to match different knowledge domains
- Expand abbreviations and domain-specific terms

**Keywords:** variant, alternative, rephrase, reword, perspective, angle, viewpoint, semantic, meaning, synonym, related, similar, equivalent, interpretation, context, expansion, broader, narrower, specific, general, aspect, dimension

**Transformation Techniques:**
1. Synonym Substitution
   - Replace key terms with synonyms
   - Example: "impact of climate change" → "effects of global warming"

2. Perspective Shifting
   - Change viewpoint or stance
   - Example: "benefits of remote work" →
     - "advantages of working from home"
     - "challenges of remote work"
     - "remote work impact on productivity"

3. Scope Modification
   - Broaden or narrow focus
   - Example: "Python web frameworks" →
     - "Python backend development tools"
     - "Flask vs Django comparison"
     - "lightweight Python web frameworks"

4. Domain Adaptation
   - Adjust terminology for different domains
   - Example: "car efficiency" →
     - "vehicle fuel economy"
     - "automotive performance metrics"
     - "eco-friendly vehicle features"

**Example Transformations:**
Original: "How to improve team communication?"
Variants:
"Best practices for team collaboration"
"Effective methods for workplace communication"
"Tools for enhancing team information sharing"
"Overcoming communication barriers in teams"
"Team communication strategies for remote work"

**Output:**
"Variation 1"
"Variation 2"

**Notes:**
- Generate up to 3 variations.
- Variations should be as diverse as possible.
- Only attempt to generate variations for topics/subjects/questions.
- Reply "NA" if it's a command or requests.
"""


class QueryExpanderAgent(BaseTool):
    def __init__(
            self, priority: int = 1, next_func: Optional[str] = None,
            ccm: ChatCompletionModel = ChatCompletionModel(),
    ):
        super().__init__(
            tool_name="QueryExpanderAgent",
            description=QUERY_EXPANDER_AGENT_PROMPT,
            priority=priority,
            next_func=next_func,
        )
        self.__ccm = ccm

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
        output = ToolOutput(tool_name=self.name, result=[])
        try:
            llm_results = self.__ccm.generate(self.description, p_query)
            for i, llm_result in enumerate(llm_results):
                sub_results = llm_result.split("\n")
                for r in sub_results:
                    if len(r) == 0 or r == "NA":
                        continue
                    item = ToolOutputItem(
                        identifier=f"query_expansion_result_{i}",
                        value=r,
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

    def validate(self, params: str) -> bool:
        params = json.loads(params)
        p_query: str = params.get("query", None)
        if p_query is None:
            return False
        p_query = p_query.strip()
        condition = [len(p_query) > 0, len(p_query) <= 200]
        if not all(condition):
            return False
        return True
