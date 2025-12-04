"""
Elementary Discourse Unit (EDU) model.
"""
from pydantic import BaseModel, Field, validator
from typing import Literal

class EDUModel(BaseModel):
    """
    Elementary Discourse Unit (EDU) in a discourse dependency tree.
    
    Attributes:
        id: Unique identifier (0 is ROOT)
        text: Text content of the EDU
        parent: ID of the parent EDU (-1 for ROOT)
        relation: Discourse relation type with parent
    """
    id: int = Field(
        description="Unique identifier for the EDU, starting from 0"
    )
    text: str = Field(
        description="The text content of the EDU"
    )
    parent: int = Field(
        description="ID of the parent EDU (-1 for ROOT)"
    )
    relation: str = Field(
        description="Discourse relation type with the parent"
    )
    
    @validator('parent')
    def validate_parent(cls, v, values):
        """Ensure parent ID is valid."""
        if 'id' in values:
            current_id = values['id']
            # ROOT has parent -1
            if current_id == 0 and v != -1:
                raise ValueError("ROOT (id=0) must have parent=-1")
            # Non-root cannot be its own parent
            if current_id > 0 and v == current_id:
                raise ValueError(
                    f"EDU cannot be its own parent (id={current_id}, parent={v})"
                )
        return v
    
    @validator('relation')
    def validate_relation(cls, v, values):
        """Normalize relation type."""
        # ROOT must have relation "null"
        if 'id' in values and values['id'] == 0:
            if v.lower() != "null":
                raise ValueError("ROOT must have relation='null'")
            return "null"
        
        # Normalize relation string
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "text": "We propose a neural network approach",
                "parent": 0,
                "relation": "ROOT"
            }
        }