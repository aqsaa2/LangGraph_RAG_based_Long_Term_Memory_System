# memory_graph/tools.py (New File)
from pydantic import BaseModel, Field

class MemoryAssessment(BaseModel):
    """Assessment of whether new, significant, or actionable information is present."""
    should_save_memory: bool = Field(
        description="True if the conversation contains truly new, significant, or actionable information that should be saved as a long-term memory. False otherwise."
    )
    reason: str = Field(
        description="A brief explanation of why memory should or should not be saved."
    )