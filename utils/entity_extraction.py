import re
import json
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from typing import List, Dict, Optional
from tqdm import tqdm

from utils.prompts.entity_extraction_prompts import entity_extraction_prompt

class Entity(BaseModel):
    name: str
    type: Optional[str] = None
    context: str
    

def extract_entities_from_paragraphs(paragraphs: List[str],
                                     relevant_entity_types: List[str] = [],
                                     usecase_context: str = "",
                                     model_name: str = "deepseek-r1:1.5b",
                                     verbose: bool = False
                                     ) -> List[Entity]:
    
    """
        Extract entities from given paragraphs, 
    """

    paragraphs = paragraphs[:25]

    # Set up some variables
    entities: List[Entity] = []
    entity_extract_llm = OllamaLLM(model=model_name, temperature=0.1)

    # Set the example entity types based on the context or the relevant entity types.
    if not relevant_entity_types and usecase_context:
        example_entity_types = f"Example entity types should adjust to this context: {usecase_context}"
    elif relevant_entity_types:
        example_entity_types = f"Example entity types: {','.join(relevant_entity_types)}"
    elif not relevant_entity_types and not usecase_context:
        example_entity_types = "There are no example entity types to adjust to. Feel free to extract any entities."
        

    # Define the pattern to extract the types from the response.
    open_think_tag, close_think_tag = "<think>", "</think>"
    open_entity_tag, close_entity_tag = "<entities>", "</entities>"
    pattern_think = re.compile(f"{open_think_tag}([\s\S]*?){close_think_tag}")
    pattern_entities = re.compile(f"{open_entity_tag}([\s\S]*?){close_entity_tag}")

    # Initialize the progress bar
    progress_bar = tqdm(total=len(paragraphs), desc="Processing paragraphs")
    entity_count = 0

    for paragraph in paragraphs:
        response = (entity_extraction_prompt | entity_extract_llm).invoke({
            "paragraph_text": paragraph,
            "example_entity_types": example_entity_types
        }).strip()
        
        if verbose:
            tqdm.write(f"Response from entity extraction: {response}")
        
        # clean the response
        if open_think_tag in response:
            try:
                # Remove the think tags from the response
                think_str = pattern_think.search(response).group(1)
                response = response.replace(think_str, "")
            except Exception as e:
                tqdm.write(f"Error extracting the think tags from the response: {e}")

        # Extract the entities from the response
        if open_entity_tag in response:
            try:
                entities_str = pattern_entities.search(response).group(1)
                entities_list = json.loads(entities_str)
                entities.extend([Entity(**entity) for entity in entities_list])
                entity_count += len(entities_list)
            except Exception as e:
                tqdm.write(f"Error extracting entities from the response: {e}")
        
        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(entity_count=entity_count)
    
    progress_bar.close()
    return entities