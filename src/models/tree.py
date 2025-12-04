"""
Discourse Dependency Tree model.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Set
from .edu import EDUModel

class DiscourseTreeModel(BaseModel):
    """
    Discourse Dependency Tree structure.
    
    Attributes:
        edus: List of Elementary Discourse Units forming a dependency tree
    """
    edus: List[EDUModel] = Field(
        description="List of Elementary Discourse Units forming a dependency tree"
    )
    
    @validator('edus')
    def validate_tree_structure(cls, v):
        """Validate that EDUs form a valid dependency tree."""
        if not v:
            raise ValueError("EDUs list cannot be empty")
        
        # Check for exactly one ROOT
        root_count = sum(1 for edu in v if edu.parent == -1)
        if root_count != 1:
            raise ValueError(f"Must have exactly one ROOT EDU, found {root_count}")
        
        # Check ROOT is at index 0
        if v[0].id != 0 or v[0].parent != -1:
            raise ValueError("First EDU must be ROOT (id=0, parent=-1)")
        
        # Check IDs are consecutive
        ids = sorted([edu.id for edu in v])
        expected_ids = list(range(len(v)))
        if ids != expected_ids:
            raise ValueError(
                f"EDU IDs must be consecutive from 0 to {len(v)-1}, got {ids}"
            )
        
        # Check for cycles
        edu_dict = {edu.id: edu.parent for edu in v}
        for edu_id in edu_dict:
            if edu_id == 0:  # Skip ROOT
                continue
            
            visited: Set[int] = set()
            current = edu_id
            
            while current != -1:
                if current in visited:
                    raise ValueError(
                        f"Circular dependency detected starting at EDU {edu_id}"
                    )
                visited.add(current)
                current = edu_dict.get(current, -1)
                
                # Safety check for invalid parent references
                if current != -1 and current not in edu_dict:
                    raise ValueError(
                        f"EDU {edu_id} references non-existent parent {current}"
                    )
        
        return v
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "edus": [edu.dict() for edu in self.edus]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DiscourseTreeModel":
        """Create from dictionary format."""
        if "edus" in data:
            return cls(edus=data["edus"])
        elif "root" in data:  # SciDTB format
            return cls.from_scidtb(data)
        else:
            raise ValueError("Invalid data format")
    
    @classmethod
    def from_scidtb(cls, data: Dict) -> "DiscourseTreeModel":
        """
        Create from SciDTB format.
        
        Args:
            data: Dictionary with 'root' key containing list of EDU dicts
            
        Returns:
            DiscourseTreeModel instance
        """
        edus = []
        for edu_data in data['root']:
            edus.append(EDUModel(
                id=edu_data['id'],
                text=edu_data['text'],
                parent=edu_data['parent'],
                relation=edu_data['relation']
            ))
        return cls(edus=edus)
    
    def get_edu_by_id(self, edu_id: int) -> Optional[EDUModel]:
        """Get EDU by ID."""
        for edu in self.edus:
            if edu.id == edu_id:
                return edu
        return None
    
    def get_children(self, parent_id: int) -> List[EDUModel]:
        """Get all children of a given EDU."""
        return [edu for edu in self.edus if edu.parent == parent_id]
    
    def get_depth(self, edu_id: int) -> int:
        """Get depth of an EDU in the tree."""
        depth = 0
        current = self.get_edu_by_id(edu_id)
        
        while current and current.parent != -1:
            depth += 1
            current = self.get_edu_by_id(current.parent)
        
        return depth
    
    def get_statistics(self) -> Dict:
        """Get tree statistics."""
        depths = [self.get_depth(edu.id) for edu in self.edus if edu.id > 0]
        relations = [edu.relation for edu in self.edus if edu.id > 0]
        
        return {
            "num_edus": len(self.edus),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "unique_relations": len(set(relations)),
            "relation_distribution": {
                rel: relations.count(rel) for rel in set(relations)
            }
        }
    
    class Config:
        schema_extra = {
            "example": {
                "edus": [
                    {
                        "id": 0,
                        "text": "ROOT",
                        "parent": -1,
                        "relation": "null"
                    },
                    {
                        "id": 1,
                        "text": "We propose a neural network approach",
                        "parent": 0,
                        "relation": "ROOT"
                    },
                    {
                        "id": 2,
                        "text": "to benefit from non-linearity",
                        "parent": 1,
                        "relation": "Enablement"
                    }
                ]
            }
        }