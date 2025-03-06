from pydantic   import BaseModel
from typing     import List, Dict, Optional, Union

class Entity(BaseModel):
    name: str
    type: Optional[str] = None
    context: str

class Relation(BaseModel):
    source_entity: Entity
    target_entity: Entity
    relation_detail: str

    def model_dump(self) -> Dict[str, Union[Dict, str]]:
        """ Dump the relation as a dictionary. """
        return {
            "source_entity": self.source_entity.model_dump(),
            "target_entity": self.target_entity.model_dump(),
            "relation_detail": self.relation_detail,
        }