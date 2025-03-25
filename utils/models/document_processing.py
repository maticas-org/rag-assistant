from pydantic                       import BaseModel
from typing                         import List, Dict, Optional, Union
from utils.models.graphrag_models   import Entity, Relation

# ------------------------------
#           Metadata
# ------------------------------

class GroupingInfo(BaseModel):
    """
        The grouping info is a dictionary that contains the information of the grouping strategy used.
        It is used to store the information of the grouping strategy used in the grouping process.
    """
    strategy:       str
    timestamp:      str
    avg_chunk_len:  int

class SummaryInfo(BaseModel):
    """
        The summary info is a dictionary that contains the information of the summarization strategy used.
        It is used to store the information of the summarization strategy used in the summarization process.
    """
    strategy:       str
    timestamp:      str
    summary_len:    int

class EntityInfo(BaseModel):
    """
        The entity info is a dictionary that contains the information of the entities extracted from the processed file.
        It is used to store the information of the entities extracted from the processed file.
    """
    number_of_entities: int
    timestamp:          str

class RelationInfo(BaseModel):
    """
        The relation info is a dictionary that contains the information of the relations extracted from the processed file.
        It is used to store the information of the relations extracted from the processed file.
    """
    number_of_relations:    List[int]
    invalid_relations:      List[int]
    timestamp:              List[str]

class Metadata(BaseModel):
    """
        The metadata is a dictionary that contains the information of the processed file.
    """
    file_hash:                  str
    processing_date:            str
    original_full_file_path:    str
    grouping_info:              Optional[GroupingInfo]
    summary_info:               Optional[SummaryInfo]
    relation_info:              Optional[RelationInfo]

# ------------------------------
#          Base File
# ------------------------------

class Paragraph(BaseModel):
    text:       str
    type:       Optional[str]
    chunk_id:   Optional[str]
    entities:   Optional[List[Entity]]
    relations:  Optional[List[Relation]]

class BaseFile(BaseModel):
    """
        The processed file refer to the first output of the pipeline, which is the result of the OCR process,
        that grabs the text and tables from the files and stores them in a structured way.
    """
    paragraphs: List[Paragraph]
    tables:     List[str]
    metadata:   Metadata
    summary:    Optional[str]