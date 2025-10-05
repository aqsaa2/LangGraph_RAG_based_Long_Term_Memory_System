# chatbot/state.py
"""Define the state structures for the chatbot."""

from typing import Optional, List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

class User(BaseModel):
    """Represents a user with their details."""
    
    user_id: Optional[str] = None
    
    def is_registered(self) -> bool:
        """Check if user has registration details."""
        return self.user_id is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "user_id": self.user_id
        }
    
    @classmethod
    def from_api_response(cls, user_details: dict) -> "User":
        """Create User from API response."""
        return cls(
            user_id=user_details.get("user_id")
        )

# Define the state of our graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    """The messages in the conversation."""
    user: Optional[User] = Field(default_factory=User) # Add the User object to state
    """The user's profile information."""

# don't strictly need KnowledgeSource or KnowledgeSearchResult for this memory system,
# but if will  plan to integrate RAG with an external knowledge base later, they can be useful.
# For now, we'll keep them out to focus on the core request.