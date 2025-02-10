import re
import json
from typing import List
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM

from utils.prompts.entity_extraction_prompts import (entity_extraction_prompt,
                                                     missing_entity_check_prompt,
                                                     additional_entity_extraction_prompt)

# Define an Entity model.
class Entity(BaseModel):
    name: str
    type: str
    context: str

def extract_entities(paragraphs:      List[str],
                     model_name:      str = "llama3.2:3b", 
                     max_llm_retries: int = 2) -> List[Entity]:
    """
    Extracts entities from a list of paragraphs. It first performs an initial extraction,
    then uses a gleaning mechanism to check for and extract any additional entities up to MAX_TIMES iterations.
    """
    llm = OllamaLLM(model=model_name, temperature=0.1)
    all_entities: List[Entity] = []

    for para in paragraphs:
        # --- Step 1: Initial extraction ---
        response = (entity_extraction_prompt | llm).invoke({
            "paragraph_text": para
        }).strip()

        current_entities: List[Entity] = []
        if "<entities>" in response:
            try:
                print(response)
                entities_str = re.search(r'<entities>([\s\S]*?)</entities>', response).group(1)
                extracted = json.loads(entities_str)
                current_entities = [
                    Entity(**entity) for entity in extracted
                    if all(key in entity for key in Entity.__fields__.keys())
                ]
            except Exception as e:
                print("Error parsing initial entities:", e)

        # --- Step 2: Gleaning mechanism (up to MAX_TIMES iterations) ---
        iteration = 0
        while iteration < max_llm_retries:

            # Create a comma-separated string of the already extracted entity names.
            entity_names = ", ".join([entity.name for entity in current_entities])
            check_response = (missing_entity_check_prompt | llm).invoke({
                "paragraph_text": para,
                "entity_names": entity_names
            }).strip()

            # Extract the response between <response> tags.
            answer = "no"
            if "<response>" in check_response:
                match = re.search(r'<response>(.*?)</response>', check_response, re.DOTALL)
                if match:
                    answer = match.group(1).strip().lower()

            if answer != "yes":
                break  # No additional entities needed.

            additional_response = (additional_entity_extraction_prompt | llm).invoke({
                "paragraph_text": para,
                "entity_names": entity_names
            }).strip()

            if "<entities>" in additional_response:
                try:
                    entities_str = re.search(r'<entities>([\s\S]*?)</entities>', additional_response).group(1)
                    new_entities_raw = json.loads(entities_str)
                    new_entities = []
                    for entity in new_entities_raw:
                        if all(key in entity for key in Entity.__fields__.keys()):
                            # Only add if this entity hasn't been extracted already.
                            if not any(existing.name == entity["name"] for existing in current_entities):
                                new_entities.append(Entity(**entity))
                    if not new_entities:
                        break  # No new entities found.
                    current_entities.extend(new_entities)
                except Exception as e:
                    print("Error parsing additional entities:", e)
            iteration += 1

        if current_entities:
            all_entities.extend(current_entities)
    
    return all_entities
