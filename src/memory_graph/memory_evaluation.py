# memory_graph/memory_evaluation.py (NEW FILE/CONTENT)

from typing import Literal, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the structured output for memory evaluation
class MemoryEvaluation(BaseModel):
    """
    Decision on whether to extract and save memory based on recent conversation.
    """
    should_save_memory: bool = Field(
        ...,
        description="True if the conversation contains new, significant, or actionable information that should be extracted and saved as a long-term memory. False otherwise."
    )
    reason: Optional[str] = Field(
        None,
        description="Brief explanation for the decision (e.g., 'user stated their name', 'user asked to be reminded of a task')."
    )
    suggested_memory_type: Optional[Literal["User", "Note", "Action", "Procedural", "Episode"]] = Field(
        None,
        description="If memory should be saved, suggest the most relevant memory type (e.g., 'User' for personal info, 'Note' for facts, 'Action' for tasks)."
    )

# System prompt for the memory evaluator LLM
EVALUATOR_SYSTEM_PROMPT = """
You are a highly discerning memory evaluator assistant. Your task is to analyze the provided conversation history and determine if there is any new, significant, or actionable information that warrants being saved as a long-term memory for the user.

Focus on:
- User's personal information (name, age, interests, occupation, preferences).
- Important facts or statements made by the user that are worth remembering.
- Specific tasks, reminders, or requests made by the user.
- Procedural knowledge (how to do something, step-by-step instructions).
- Key events or episodes in the conversation that are distinct and valuable to recall later.

Do NOT save:
- Trivial conversational turns (greetings, small talk, acknowledgements).
- Information already explicitly known or previously stated (unless it's an update).
- Information that is temporary or context-specific to the immediate turn but not for long-term recall.
- Repetitive statements.

Provide your decision and a brief reason. If you decide to save memory, also suggest the most appropriate memory type.

Conversation history:
{conversation_history}
"""

def create_memory_evaluator(model_name: str):
    """Creates a LangChain Runnable for memory evaluation."""
    from langchain_core.runnables import RunnableLambda
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_google_vertexai import ChatVertexAI
    from langchain.output_parsers.pydantic import PydanticOutputParser

    llm = ChatVertexAI(model_name=model_name, temperature=0) # Low temperature for consistent output

    parser = PydanticOutputParser(pydantic_object=MemoryEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # This chain will output an AIMessage with a tool call conforming to MemoryEvaluation
    # It might be more robust to directly parse the output to Pydantic object
    chain = (
        prompt
        | llm.with_structured_output(MemoryEvaluation) # Ensure structured output
    )
    return chain