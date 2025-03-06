import re
import json
from datetime import datetime

from typing             import List, Dict, Optional, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse
from tqdm               import tqdm

from utils.models.graphrag_models                   import Entity, Relation
from utils.prompts.graphrag.relationship_extraction_prompts import (relationship_extraction_prompt,
                                                            missing_relations_check_prompt,
                                                            additional_relations_extraction_prompt)

def clean_response(response: str, verbose: bool = False) -> str:
    """
    Clean the response by completely removing the <think>...</think> section.
    """
    open_think_tag, close_think_tag = "<think>", "</think>"
    pattern_think = re.compile(rf"{open_think_tag}.*?{close_think_tag}", re.DOTALL)

    # Ensure we remove the entire section, including the tags
    response = pattern_think.sub("", response).strip()

    if verbose:
        print("Cleaned Response:", response)  # Debugging line to inspect output

    return response


def extract_relations_from_response(response: str,
                                    entities: List[Entity],
                                    verbose: bool = False) -> List[Relation]:
    """
    Extract the relations from the response.
    """
    # Set up some variables
    relationships: List[Relation] = []

    # Define the pattern to extract the relationships section
    open_relation_tag, close_relation_tag = "<relationships>", "</relationships>"
    pattern_relationship = re.compile(rf"{open_relation_tag}(.*?){close_relation_tag}", re.DOTALL)

    # Clean the response
    response = clean_response(response, verbose)

    # Extract the relationships JSON
    match = pattern_relationship.search(response)
    if match:
        relationship_str = match.group(1).strip()

        try:
            relationship_list = json.loads(relationship_str)
            relationship_list = validate_relations(entities, relationship_list, verbose)
            relationships.extend([Relation(**r) for r in relationship_list])
        except json.JSONDecodeError as e:
            if verbose:
                tqdm.write(f"Error parsing JSON from the response: {e}")
                tqdm.write(f"Raw extracted JSON string: {relationship_str}")
    else:
        if verbose:
            tqdm.write("No relationships found in the response.")

    return relationships


def validate_relations(entities: List[Entity],
                       relations: List[Dict],
                       verbose: bool = False) -> List[Relation]:
    """
        Validate the relations extracted from the response.
        Relations should only contain entities that are present in the entities list.
    """
    valid_relations = []
    entity_names = [entity.name for entity in entities]

    for relation in relations:
        if relation.get("source_entity") in entity_names and relation.get("target_entity") in entity_names:
            valid_relations.append(relation)
        else:
            if verbose:
                tqdm.write(f"Invalid relation: {relation}")

    # Map the source and target entities to their respective objects
    for r in valid_relations:
        r["source_entity"] = next(e for e in entities if e.name == r["source_entity"])
        r["target_entity"] = next(e for e in entities if e.name == r["target_entity"])
    
    return valid_relations


def extract_relations_from_paragraphs(
                                        llm:                    Union[ChatOllama, ChatBedrockConverse],                 
                                        paragraphs:             List[Dict[str, str]],
                                        verbose:                bool = False
                                     ) -> Dict[str, List[Relation]]:
    
    """
        Extract relationships among entities from given paragraphs using the provided language model,
        in combination with the original paragraphs and entities extracted from them.

        Args:
            - llm: Union[ChatOllama, ChatBedrockConverse] - The language model to use for entity extraction.
            - paragraphs: List[Dict[str, str]] - The grouped paragraphs to extract relationships from.
            - verbose: bool - Whether to print the progress.
    """

    # Initialize the progress bar
    progress_bar = tqdm(total=len(paragraphs), desc="Extracting relations from paragraph")
    relations_count = 0
    invalid_relations = 0

    for p in paragraphs:
        ptext = p["text"]
        entities = [Entity(**e) for e in p["entities"]]

        response = (relationship_extraction_prompt | llm).invoke({
            "paragraph_text": ptext,
            "found_entities": entities,
        }).content.strip()
        
        print(response)
        
        # Clean and extract the entities from the response
        relations = extract_relations_from_response(response, entities, verbose)
        relations = [r.model_dump() for r in relations]
        print("Extracted Relations: ", relations)

        p["relations"] = relations
        p["relations_metadata"] = {
            "relations_extraction_timestamp": datetime.now().isoformat(),
            "invalid_relations": invalid_relations,
            "total_relations": len(relations),
        }

        # Update the progress bar
        relations_count += len(relations)
        progress_bar.update(1)
        progress_bar.set_postfix(rel_count=relations_count)
        progress_bar.refresh()  

        # Only once iter for testing
        # break
        
    progress_bar.close()

    return paragraphs
