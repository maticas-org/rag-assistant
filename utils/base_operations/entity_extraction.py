import re
import json
from datetime import datetime

from typing             import List, Dict, Optional, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse
from tqdm               import tqdm

from utils.models.graphrag_models import Entity
from utils.prompts.graphrag.entity_extraction_prompts import entity_extraction_prompt
    

def clean_response(response: str, verbose: bool = False) -> str:
    """
    Clean the response by removing the think tags.
    """
    open_think_tag, close_think_tag = "<think>", "</think>"
    pattern_think = re.compile(f"{open_think_tag}([\s\S]*?){close_think_tag}")
    
    if open_think_tag in response:
        try:
            # Remove the think tags from the response
            think_str = pattern_think.search(response).group(1)
            response = response.replace(think_str, "")
        except Exception as e:
            if verbose:
                tqdm.write(f"Error extracting the think tags from the response: {e}")
    
    return response

def extract_entities_from_response(response: str, verbose: bool = False) -> List[Entity]:
    """
    Extract entities from the response.
    """

    # Set up some variables
    entities: List[Entity] = []

    # Define the pattern to extract the types from the response.
    open_entity_tag, close_entity_tag = "<entities>", "</entities>"
    pattern_entities = re.compile(f"{open_entity_tag}([\s\S]*?){close_entity_tag}")

    # clean the response
    response = clean_response(response, verbose)

    # Extract the entities from the response
    if open_entity_tag in response:
        try:
            entities_str = pattern_entities.search(response).group(1)
            entities_list = json.loads(entities_str)
            entities.extend([Entity(**entity) for entity in entities_list])
        except Exception as e:
            if verbose:
                tqdm.write(f"Error extracting entities from the response: {e}")
    
    return entities

def extract_entities_from_paragraphs(
                                        llm:                    Union[ChatOllama, ChatBedrockConverse],                 
                                        paragraphs:             List[Dict[str, str]],
                                        relevant_entity_types:  List[str] = [],
                                        usecase_context:        str = "",
                                        verbose:                bool = False
                                     ) -> Dict[str, List[Entity]]:
    
    """
        Extract entities from given paragraphs using the provided language model as well as the relevant entity types,
        or the usecase context if available.

        Args:
            - llm: Union[ChatOllama, ChatBedrockConverse] - The language model to use for entity extraction.
            - paragraphs: List[str] - The paragraphs to extract entities from.
            - relevant_entity_types: List[str] - The relevant entity types to consider.
            - usecase_context: str - The context of the usecase.
            - verbose: bool - Whether to print the progress.
    """

    # Set the example entity types based on the context or the relevant entity types.
    if not relevant_entity_types and usecase_context:
        example_entity_types = f"Example entity types should adjust to this context: {usecase_context}"
    elif relevant_entity_types:
        example_entity_types = f"Example entity types: {','.join(relevant_entity_types)}"
    elif not relevant_entity_types and not usecase_context:
        example_entity_types = "There are no example entity types to adjust to. Feel free to extract any entities."

    # Initialize the progress bar
    progress_bar = tqdm(total=len(paragraphs), desc="Extracting Entities from paragraph")
    entity_count = 0

    for p in paragraphs:
        ptext = p["text"]

        response = (entity_extraction_prompt | llm).invoke({
            "paragraph_text": ptext,
            "example_entity_types": example_entity_types
        }).content.strip()
        
        #if verbose:
        #    tqdm.write(f"Response from entity extraction: {response}")
        
        # Clean and extract the entities from the response
        entities_ = extract_entities_from_response(response, verbose)
        entities_ = [entity.model_dump() for entity in entities_]
        p["entities"] = entities_


        # Update the progress bar
        entity_count += len(entities_)
        progress_bar.update(1)
        progress_bar.set_postfix(entity_count=entity_count)
    
    progress_bar.close()

    return paragraphs